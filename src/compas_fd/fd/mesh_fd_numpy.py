from numpy import array
from numpy import float64

from compas_fd.fd import fd_numpy
# from compas_fd.loads import SelfweightCalculator


def mesh_fd_numpy(mesh):
    """Find the equilibrium shape of a mesh for the given force densities.

    Parameters
    ----------
    mesh : :class:`compas_fd.datastructures.CableMesh`
        The mesh to equilibriate.

    Returns
    -------
    :class:`compas_fd.datastructures.CableMesh`
        The function returns an updated mesh.

    """
    k_i = mesh.key_index()
    fixed = mesh.vertices_where({'is_anchor': True})
    fixed = [k_i[key] for key in fixed]
    vertices = array(mesh.vertices_attributes('xyz'), dtype=float64)
    loads = array(mesh.vertices_attributes(('px', 'py', 'pz')), dtype=float64)
    edges = [(k_i[u], k_i[v]) for u, v in mesh.edges_where({'is_edge': True})]
    q = array([attr['q'] for key, attr in mesh.edges_where({'is_edge': True}, True)], dtype=float64).reshape((-1, 1))

    # density = mesh.attributes['density']
    # calculate_sw = SelfweightCalculator(mesh, density=density)

    vertices, r, f = fd_numpy(vertices=vertices, edges=edges, loads=loads, q=q, fixed=fixed)

    for key, attr in mesh.vertices(True):
        index = k_i[key]
        attr['x'] = vertices[index, 0]
        attr['y'] = vertices[index, 1]
        attr['z'] = vertices[index, 2]
        attr['_rx'] = r[index, 0]
        attr['_ry'] = r[index, 1]
        attr['_rz'] = r[index, 2]

    for index, (key, attr) in enumerate(mesh.edges_where({'_is_edge': True}, True)):
        attr['_f'] = f[index, 0]
