from typing import List, Tuple
from typing_extensions import Annotated

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from compas.numerical import connectivity_matrix
from compas.numerical import normrow

# from compas_fd.loads import SelfweightCalculator


Point = Annotated[List[float], 3]
Load = Annotated[List[float], 3]
Force = Annotated[List[float], 3]


def fd_numpy(*,
             vertices: List[Point],
             edges: List[Tuple[int, int]],
             loads: List[Load],
             q: List[float],
             fixed: List[int]) -> Tuple[List[Point], List[float], List[Force]]:
    """Implementation of the force density method to compute equilibrium of axial force networks.
    Parameters
    ----------
    vertices : list
        XYZ coordinates of the vertices of the network
    edges : list
        Edges between vertices represented by
    fixed : list
        Indices of fixed vertices.
    q : list
        Force density of edges.
    loads : list
        XYZ components of the loads on the vertices.
    Returns
    -------
    vertices : array
        XYZ coordinates of the equilibrium geometry.
    r : array
        Residual forces.
    f : array
        Forces in the edges.
    Notes
    -----
    For more info, see [1]_
    References
    ----------
    .. [1] Schek H., *The Force Density Method for Form Finding and Computation of General Networks*,
           Computer Methods in Applied Mechanics and Engineering 3: 115-134, 1974.
    """
    free = list(set(range(len(vertices)) - set(fixed)))

    # density = mesh.attributes['density']
    # calculate_sw = SelfweightCalculator(mesh, density=density)

    # if density:
    #     sw = calculate_sw(vertices)
    #     p[:, 2] = -sw[:, 0]

    C = connectivity_matrix(edges, 'csr')
    Ci = C[:, free]
    Cf = C[:, fixed]
    Ct = C.transpose()
    Cit = Ci.transpose()
    Q = diags([q.flatten()], [0])
    A = Cit.dot(Q).dot(Ci)
    b = loads[free] - Cit.dot(Q).dot(Cf).dot(vertices[fixed])

    vertices[free] = spsolve(A, b)

    # if density:
    #     sw = calculate_sw(vertices)
    #     p[:, 2] = -sw[:, 0]

    l = normrow(C.dot(vertices))  # noqa: E741
    f = q * l
    r = loads - Ct.dot(Q).dot(C).dot(vertices)

    return vertices, r, f
