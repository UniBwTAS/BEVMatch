"""Architekturvarianten fuer die direkte SE(2)-Pose-Regression auf eingefrorenen BEV-Features.

Warum ueberhaupt neue Varianten? Der fruehere Kopf (the naive concat-CNN head) ist auf
den Datensatz-Mittelwert kollabiert: Vorhersage-Streuung 0.005 m bei GT-Streuung 3.07 m, RTE
2.648 m gegen 2.633 m des Null-Praediktors. Er hat die Feature-Karte mit 4x stride-2 auf 12x12
reduziert -- bei 0.6 m Zellgroesse sind das ~9 m pro Zelle, also faellt eine close-Verschiebung
(0-5 m) unter die Aufloesung, bevor das MLP sie sehen kann. Ausserdem musste er die Verschiebung
implizit aus konkatenierten Feature-Karten erschliessen, ohne jede geometrische Struktur.

Die Varianten hier setzen genau dort an:

  corr     Cost-Volume: normalisierte Kreuzkorrelation zwischen f0 und allen ganzzahlig
           verschobenen Varianten von f1. Die Verschiebung ist damit eine ACHSE des Tensors
           statt einer zu erschliessenden Groesse; Soft-Argmax liest sie sub-zellgenau ab.
           Das ist die geometrisch ehrlichste Formulierung und die mit der besten Aussicht.
  corrcnn  wie corr, aber das Cost-Volume wird von einem kleinen CNN gelesen statt per
           Soft-Argmax. Robuster gegen mehrdeutige Korrelationsmaxima (Symmetrien, offenes Feld).
  hires    CNN im Stil des alten Kopfes, aber nur 2x stride-2 (45x45, ~2.4 m/Zelle) und
           CoordConv-Kanaele, damit Ortsinformation den Flatten ueberlebt.
  deep     Reimplementierung des alten Kopfes als Kontrollgruppe. Kollabiert sie erneut,
           ist der Aufloesungsverlust bestaetigt; laeuft sie diesmal, lag es am Trainingssetup.

Alle Koepfe geben (t_xy, [cos, sin]) zurueck. Rotationen zwischen zwei Frames sind klein, daher
wird der Winkel als Einheitsvektor regressiert statt als Bogenmass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _coord_channels(b, h, w, device):
    """Zwei Kanaele mit normierten x/y-Koordinaten (CoordConv)."""
    ys = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
    xs = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([ys, xs], 1)


def build_cost_volume(f0, f1, max_shift, pool=2):
    """Normalisierte Kreuzkorrelation ueber alle Verschiebungen in [-max_shift, max_shift].

    f0, f1: [B,C,H,W]. Rueckgabe: [B, D, D, ...] als [B, D*D, 1, 1]-freies Volumen der Form
    [B, D, D] mit D = 2*max_shift+1, plus die Achsenwerte in Zellen.

    Implementierung ueber F.conv2d waere schneller, ist aber fuer nicht-quadratische Ausschnitte
    fehleranfaellig; das explizite Rollen ist eindeutig und bei D<=41 schnell genug.
    """
    if pool > 1:
        # Vor der Korrelation poolen: viertelt den Speicher UND vervierfacht die abgedeckte
        # Strecke bei gleichem max_shift. Noetig, weil der far-Bin bis 20 m reicht -- das sind
        # bei 0.6 m/Zelle 33 Zellen, die max_shift=20 auf voller Aufloesung nicht erreicht.
        # Die Genauigkeit leidet nicht proportional, weil Soft-Argmax sub-zellgenau interpoliert.
        f0 = F.avg_pool2d(f0, pool)
        f1 = F.avg_pool2d(f1, pool)
    a = F.normalize(f0, dim=1)
    c = F.normalize(f1, dim=1)
    b, _, h, w = a.shape
    # Einmal nullgepolstert; jede Verschiebung ist danach ein SLICE (eine View) statt einer
    # eigenen Tensorkopie. Mit torch.roll landeten bei max_shift=20 alle (2*20+1)^2 = 1681
    # gerollten Kopien im Autograd-Graph -- bei C=512 auf 180x180 waren das >60 GB und der
    # Lauf starb an CUDA-OOM. Views kosten nichts zusaetzlich, der Backward braucht c nur einmal.
    cpad = F.pad(c, [max_shift] * 4)
    d = 2 * max_shift + 1
    rows = []
    for i in range(d):
        cols = []
        for j in range(d):
            # i=j=0 entspricht Verschiebung (-max_shift, -max_shift); das Fenster wandert mit,
            # sodass Index (i,j) die tatsaechliche Verschiebung (i-max_shift, j-max_shift) meint.
            sl = cpad[:, :, i:i + h, j:j + w]
            cols.append((a * sl).sum(dim=1).mean(dim=(1, 2)))
        rows.append(torch.stack(cols, dim=1))
    return torch.stack(rows, dim=1)                      # [B, d, d]


class CorrSoftArgmaxHead(nn.Module):
    """Cost-Volume + Soft-Argmax. Wenige Parameter, maximale geometrische Struktur."""

    def __init__(self, c_in, res_m, max_shift=20, pool=2, **kw):
        super().__init__()
        self.max_shift = max_shift
        self.pool = pool
        self.res_m = res_m * pool          # eine Volumen-Zelle entspricht pool Feature-Zellen
        self.embed = nn.Sequential(nn.Conv2d(c_in, 32, 1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 32, 3, 1, 1))
        self.temp = nn.Parameter(torch.tensor(8.0))
        self.rot = nn.Sequential(nn.Conv2d(2 * c_in, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
                                 nn.Conv2d(128, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
                                 nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(128 * 16, 128),
                                 nn.ReLU(inplace=True), nn.Linear(128, 2))

    def forward(self, f0, f1):
        e0, e1 = self.embed(f0), self.embed(f1)
        vol = build_cost_volume(e0, e1, self.max_shift, self.pool)   # [B,D,D]
        b, d, _ = vol.shape
        p = torch.softmax((vol * self.temp).reshape(b, -1), dim=1).reshape(b, d, d)
        idx = torch.arange(-self.max_shift, self.max_shift + 1, device=vol.device, dtype=vol.dtype)
        dy = (p.sum(dim=2) * idx).sum(dim=1)
        dx = (p.sum(dim=1) * idx).sum(dim=1)
        t = torch.stack([dy, dx], 1) * self.res_m
        ang = self.rot(torch.cat([f0, f1], 1))
        return t, ang / (ang.norm(dim=1, keepdim=True) + 1e-6)


class CorrCNNHead(nn.Module):
    """Cost-Volume, aber von einem CNN gelesen -- robuster bei mehrdeutigen Maxima."""

    def __init__(self, c_in, res_m, max_shift=20, pool=2, **kw):
        super().__init__()
        self.max_shift = max_shift
        self.pool = pool
        self.res_m = res_m * pool
        self.embed = nn.Sequential(nn.Conv2d(c_in, 32, 1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 32, 3, 1, 1))
        d = 2 * max_shift + 1
        self.read = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
                                  nn.Flatten(), nn.Linear(64 * d * d, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, 4))

    def forward(self, f0, f1):
        e0, e1 = self.embed(f0), self.embed(f1)
        vol = build_cost_volume(e0, e1, self.max_shift, self.pool).unsqueeze(1)  # [B,1,D,D]
        b, _, d, _ = vol.shape
        x = torch.cat([vol, _coord_channels(b, d, d, vol.device)], 1)    # Verschiebung als Ort
        o = self.read(x)
        t = o[:, :2] * self.max_shift * self.res_m
        ang = o[:, 2:4]
        return t, ang / (ang.norm(dim=1, keepdim=True) + 1e-6)


class HiResCNNHead(nn.Module):
    """Der alte Ansatz, aber mit erhaltener Aufloesung (2x statt 4x stride-2) + CoordConv."""

    def __init__(self, c_in, res_m, grid=180, **kw):
        super().__init__()
        self.res_m = res_m
        self.proj = nn.Conv2d(c_in, 48, 1)
        self.trunk = nn.Sequential(
            nn.Conv2d(2 * 48 + 2, 96, 3, 2, 1), nn.GroupNorm(8, 96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, 1), nn.ReLU(inplace=True))
        g = (grid + 1) // 2
        g = (g + 1) // 2
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(32 * g * g, 512), nn.ReLU(inplace=True),
                                 nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 4))

    def forward(self, f0, f1):
        p0, p1 = self.proj(f0), self.proj(f1)
        b, _, h, w = p0.shape
        x = torch.cat([p0, p1, _coord_channels(b, h, w, p0.device)], 1)
        o = self.mlp(self.trunk(x))
        t = o[:, :2] * 20.0
        ang = o[:, 2:4]
        return t, ang / (ang.norm(dim=1, keepdim=True) + 1e-6)


class DeepCNNHead(nn.Module):
    """Kontrollgruppe: der frueher kollabierte Kopf, unveraendert in der Struktur."""

    def __init__(self, c_in, res_m, grid=180, **kw):
        super().__init__()
        self.res_m = res_m
        self.proj = nn.Conv2d(c_in, 64, 1)
        ch = [128, 128, 192, 256]
        layers, prev = [], 128
        for c in ch:
            layers += [nn.Conv2d(prev, c, 3, 2, 1), nn.GroupNorm(8, c), nn.ReLU(inplace=True)]
            prev = c
        layers += [nn.Conv2d(prev, 64, 1), nn.ReLU(inplace=True)]
        self.trunk = nn.Sequential(*layers)
        g = grid
        for _ in ch:
            g = (g + 1) // 2
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(64 * g * g, 512), nn.ReLU(inplace=True),
                                 nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 4))

    def forward(self, f0, f1):
        x = torch.cat([self.proj(f0), self.proj(f1)], 1)
        o = self.mlp(self.trunk(x))
        ang = o[:, 2:4]
        return o[:, :2], ang / (ang.norm(dim=1, keepdim=True) + 1e-6)


HEADS = {
    'corr': CorrSoftArgmaxHead,
    'corrcnn': CorrCNNHead,
    'hires': HiResCNNHead,
    'deep': DeepCNNHead,
}


def build_head(name, c_in, res_m, grid, max_shift=20, **kw):
    return HEADS[name](c_in=c_in, res_m=res_m, grid=grid, max_shift=max_shift, **kw)


def _rotate(x, deg):
    """Dreht [B,C,H,W] um das Kartenzentrum (deg als Skalar-Tensor oder float)."""
    b = x.shape[0]
    th = torch.deg2rad(torch.as_tensor(deg, dtype=x.dtype, device=x.device)).reshape(1).expand(b)
    cos, sin = torch.cos(th), torch.sin(th)
    mat = x.new_zeros(b, 2, 3)
    mat[:, 0, 0], mat[:, 0, 1] = cos, -sin
    mat[:, 1, 0], mat[:, 1, 1] = sin, cos
    grid = F.affine_grid(mat, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')


class CorrSE2Head(nn.Module):
    """3D-Cost-Volume ueber (Winkel, dy, dx) -- schaetzt die SE(2)-Pose als Ganzes.

    Warum nicht getrennt? Der erste Entwurf hat die Translation aus einem Cost-Volume gelesen und
    den Winkel separat durch ein MLP geschaetzt. Das MLP kollabierte (RRE 6.78 gegen 6.51 des
    Null-Praediktors) -- dieselbe Bauform, die schon bei der Translation kollabiert ist. Zudem
    sind Rotation und Translation bei SE(2) gekoppelt: eine Drehung um das Kartenzentrum
    verschiebt alles ausserhalb des Zentrums, also ist die beste Translation vom angenommenen
    Winkel abhaengig. Beide Groessen gehoeren deshalb in EIN Volumen, aus dem ein gemeinsamer
    Soft-Argmax alle drei Freiheitsgrade sub-zellgenau abliest.
    """

    def __init__(self, c_in, res_m, max_shift=20, pool=2, n_rot=9, max_deg=8.0, **kw):
        super().__init__()
        self.max_shift = max_shift
        self.pool = pool
        self.res_m = res_m * pool
        self.max_deg = max_deg
        self.register_buffer('angles', torch.linspace(-max_deg, max_deg, n_rot))
        self.embed = nn.Sequential(nn.Conv2d(c_in, 32, 1), nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 32, 3, 1, 1))
        self.temp = nn.Parameter(torch.tensor(8.0))

    def forward(self, f0, f1):
        e0, e1 = self.embed(f0), self.embed(f1)
        if self.pool > 1:                     # einmal poolen, dann rotieren: spart Rechenzeit
            e0 = F.avg_pool2d(e0, self.pool)
            e1 = F.avg_pool2d(e1, self.pool)
        # ZURUECK-Drehen um -a: ist f1 um a gedreht, richtet _rotate(f1, -a) es wieder an f0 aus,
        # und der Index a traegt damit die tatsaechliche Rotation. Mit +a faende der Kopf
        # systematisch -a; bei fest orientierter Soft-Argmax-Achse ist das nicht kompensierbar
        # und der Winkel kollabiert auf den Mittelwert (so geschehen in der Polar-Variante).
        vols = [build_cost_volume(e0, _rotate(e1, -float(a)), self.max_shift, pool=1)
                for a in self.angles]
        vol = torch.stack(vols, dim=1)                       # [B, K, D, D]
        b, k, d, _ = vol.shape
        p = torch.softmax((vol * self.temp).reshape(b, -1), dim=1).reshape(b, k, d, d)
        idx = torch.arange(-self.max_shift, self.max_shift + 1, device=vol.device, dtype=vol.dtype)
        dy = (p.sum(dim=(1, 3)) * idx).sum(dim=1)
        dx = (p.sum(dim=(1, 2)) * idx).sum(dim=1)
        deg = (p.sum(dim=(2, 3)) * self.angles.to(vol.dtype)).sum(dim=1)
        t = torch.stack([dy, dx], 1) * self.res_m
        th = torch.deg2rad(deg)
        return t, torch.stack([torch.cos(th), torch.sin(th)], 1)


HEADS['corrse2'] = CorrSE2Head
