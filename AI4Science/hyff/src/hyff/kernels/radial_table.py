"""C2 cubic spline / quintic Hermite compilation of learned radial filters.

Quintic coefficients match value and two derivatives at each knot. Cubic
splines match knot values and endpoint slopes with C2 internal continuity.
This is an approximate inference representation; its energy/force error must be
checked separately against the model with analytic radial filters. Coordinate forces
are differentiated from the interpolated energy itself. torch.compile can
fuse the coefficient gathers, polynomial evaluation and its full derivative.
"""
import torch
from torch import nn


def filter_jet(rbf, network, distance):
    """Analytic value/first/second derivative of Gaussian/cosine -> SiLU MLP.

    Compilation runs on the CPU in float64. The derivatives are derived by hand,
    not taken from autograd.
    """
    centers = rbf.centers.detach().cpu().double()
    gamma = rbf.gamma.detach().cpu().double()
    delta = distance[:, None] - centers[None]
    gaussian = torch.exp(-gamma * delta.square())
    gp = -2 * gamma * delta * gaussian
    gpp = (4 * gamma.square() * delta.square() - 2 * gamma) * gaussian
    frequency = torch.pi / rbf.cutoff
    envelope = .5 * (1 + torch.cos(frequency * distance))
    ep = -.5 * frequency * torch.sin(frequency * distance)
    epp = -.5 * frequency ** 2 * torch.cos(frequency * distance)
    x = gaussian * envelope[:, None]
    xp = gp * envelope[:, None] + gaussian * ep[:, None]
    xpp = gpp * envelope[:, None] + 2 * gp * ep[:, None] + gaussian * epp[:, None]
    for layer in network:
        if hasattr(layer, 'weight'):
            weight = layer.weight.detach().cpu().double()
            bias = layer.bias.detach().cpu().double() if layer.bias is not None else None
            x = torch.nn.functional.linear(x, weight, bias)
            xp = torch.nn.functional.linear(xp, weight)
            xpp = torch.nn.functional.linear(xpp, weight)
        elif isinstance(layer, nn.SiLU):
            sigmoid = torch.sigmoid(x)
            first = sigmoid * (1 + x * (1 - sigmoid))
            second = sigmoid * (1 - sigmoid) * (2 + x * (1 - 2 * sigmoid))
            xpp = second * xp.square() + first * xpp
            xp = first * xp
            x = x * sigmoid
        else:
            raise TypeError(f'Unsupported radial layer: {type(layer)}')
    return x, xp, xpp


class QuinticRadialTable(nn.Module):
    def __init__(self, coefficients, cutoff):
        super().__init__()
        self.register_buffer('coefficients', coefficients)
        self.intervals = len(coefficients)
        self.cutoff = float(cutoff)
        self.inverse_spacing = self.intervals / self.cutoff

    @classmethod
    def compile(cls, rbf, network, intervals=200, table_cutoff=None):
        if intervals < 2:
            raise ValueError('At least two intervals are needed')
        domain = rbf.cutoff if table_cutoff is None else float(table_cutoff)
        if not 0 < domain <= rbf.cutoff:
            raise ValueError('Table domain must lie within the radial basis domain')
        knots = torch.linspace(0., domain, intervals + 1, dtype=torch.float64)
        y, first, second = filter_jet(rbf, network, knots)
        h = domain / intervals
        a0 = y[:-1]
        a1 = h * first[:-1]
        a2 = .5 * h * h * second[:-1]
        residual = y[1:] - a0 - a1 - a2
        slope = h * first[1:] - a1 - 2 * a2
        curvature = h * h * second[1:] - 2 * a2
        coefficients = torch.stack((a0, a1, a2, 10 * residual - 4 * slope + .5 * curvature,
                                    -15 * residual + 7 * slope - curvature,
                                    6 * residual - 3 * slope + .5 * curvature), dim=1)
        return cls(coefficients, domain)

    def forward(self, distance):
        coordinate = (distance * self.inverse_spacing).clamp(0., float(self.intervals))
        index = coordinate.floor().long().clamp(max=self.intervals - 1)
        x = (coordinate - index.to(coordinate.dtype))[:, None]
        degree = self.coefficients.shape[1] - 1
        value = self.coefficients[index, degree]
        for power in range(degree - 1, -1, -1):
            value = value * x + self.coefficients[index, power]
        return value


class CubicRadialTable(QuinticRadialTable):
    """Uniform C2 cubic spline with independently known endpoint slopes.

    Unlike a per-interval cubic Hermite fit, the internal derivatives are
    coupled by a tridiagonal solve. This preserves C2 continuity while storing
    four coefficients per interval. Learned parameters are never changed.
    """

    @classmethod
    def compile(cls, rbf, network, intervals=400, table_cutoff=None):
        if intervals < 2:
            raise ValueError('At least two intervals are needed')
        domain = rbf.cutoff if table_cutoff is None else float(table_cutoff)
        if not 0 < domain <= rbf.cutoff:
            raise ValueError('Table domain must lie within the radial basis domain')
        knots = torch.linspace(0., domain, intervals + 1, dtype=torch.float64)
        values, first, _ = filter_jet(rbf, network, knots)
        h = domain / intervals
        rhs = torch.empty_like(values)
        rhs[0] = 6 / h * ((values[1] - values[0]) / h - first[0])
        rhs[-1] = 6 / h * (first[-1] - (values[-1] - values[-2]) / h)
        rhs[1:-1] = 6 / h**2 * (values[2:] - 2 * values[1:-1] + values[:-2])
        diagonal = torch.full((intervals + 1,), 4., dtype=torch.float64)
        diagonal[0] = diagonal[-1] = 2.
        # Thomas elimination is O(intervals * output channels), on the CPU.
        for i in range(1, intervals + 1):
            multiplier = 1 / diagonal[i - 1]
            diagonal[i] -= multiplier
            rhs[i] -= multiplier * rhs[i - 1]
        second = torch.empty_like(rhs)
        second[-1] = rhs[-1] / diagonal[-1]
        for i in range(intervals - 1, -1, -1):
            second[i] = (rhs[i] - second[i + 1]) / diagonal[i]
        a0 = values[:-1]
        a1 = values[1:] - values[:-1] - h**2 / 6 * (2 * second[:-1] + second[1:])
        a2 = .5 * h**2 * second[:-1]
        a3 = h**2 / 6 * (second[1:] - second[:-1])
        return cls(torch.stack((a0, a1, a2, a3), dim=1), domain)
