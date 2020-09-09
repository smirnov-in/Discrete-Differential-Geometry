import numpy as np


class Surface:
    def __init__(self, faces):
        self.vertices = np.unique(faces)
        self.faces = np.array(faces)
        self.edges = np.unique(  # неориентированные рёбра
            [np.sort([face[i], face[i + 1]]) for face in faces for i in np.arange(-1, 2)], axis=0)

    def _get_edges(self, face):
        if face in self.faces:
            return np.array([np.sort([face[i], face[i + 1]]) for i in np.arange(-1, 2)])

    def _get_vertex_neighbours(self, vertex):
        neighbours = []

        for edge in self.edges:
            if vertex in edge:
                neighbours.append(edge)

        neighbours = list(np.unique(neighbours))
        neighbours.remove(vertex)

        return np.array(neighbours)

    def _get_face_neighbours(self, face):
        neighbours = []

        for edge_1 in self._get_edges(face):
            for other_face in self.faces:
                for edge_2 in self._get_edges(other_face):
                    if (edge_1 == edge_2).all():
                        neighbours.append(other_face)

        neighbours = np.unique(neighbours, axis=0)

        for i in np.arange(len(neighbours)):
            if (neighbours[i] == face).all():
                return np.delete(neighbours, i, axis=0)

    def is_surface(self):
        for face in self.faces:  # проверка, что все вершины грани различны
            if len(set(face)) != 3:
                return False

        for i in np.arange(len(self.faces)):  # проверка, что все грани различны (как множества)
            for j in np.arange(len(self.faces)):
                if i != j and set(self.faces[i]) == set(self.faces[j]):
                    return False

        for edge_1 in self.edges:  # проверка, что каждое ребро ровно в 2-х гранях
            count = 0

            for face in self.faces:
                for edge_2 in self._get_edges(face):
                    if (edge_1 == edge_2).all():
                        count += 1

            if count != 2:
                return False

        for vertex in self.vertices:  # проверка, что окрестность каждое точки – это диск
            neighbours = self._get_vertex_neighbours(vertex)
            used = {neighbour: False for neighbour in neighbours}
            used[neighbours[0]] = True
            queue = [neighbours[0]]

            while len(queue) != 0:
                neighbour = queue.pop(0)

                for face in self.faces:
                    for i in np.arange(3):
                        for j in np.arange(3):
                            if i != j and vertex == face[i] and neighbour == face[j]:
                                if not used[face[3 - i - j]]:
                                    used[face[3 - i - j]] = True
                                    queue.append(face[3 - i - j])

            if False in used.values():
                return False

        return True

    def is_connected(self):
        used = {vertex: False for vertex in self.vertices}
        used[self.vertices[0]] = True
        queue = [self.vertices[0]]

        while len(queue) != 0:
            vertex = queue.pop(0)

            for new_vertex in self._get_vertex_neighbours(vertex):
                if not used[new_vertex]:
                    used[new_vertex] = True
                    queue.append(new_vertex)

        return not (False in used.values())

    def is_oriented(self):
        for face in self.faces:
            for neighbour in self._get_face_neighbours(face):
                for i in np.arange(3):  # находим общие вершины
                    for j in np.arange(3):
                        if face[i] == neighbour[j] and face[(i + 1) % 3] == neighbour[(j + 1) % 3]:
                            return False

        return True

    def is_orientable(self):
        used = {str(face): False for face in self.faces}
        orientation = {str(self.faces[0]): True}
        used[str(self.faces[0])] = True
        queue = [self.faces[0]]

        while len(queue) != 0:
            face = queue.pop(0)

            for neighbour in self._get_face_neighbours(face):
                for i in np.arange(3):
                    visited = False

                    for j in range(3):
                        if face[i] == neighbour[j]:
                            coorintated = (face[(i + 1) % 3] != neighbour[(j + 1) % 3])

                            if used[str(neighbour)] and (orientation[str(face)] == orientation[
                                    str(neighbour)]) != coorintated:
                                return False
                            elif not used[str(neighbour)]:
                                used[str(neighbour)] = True
                                queue.append(neighbour)
                                orientation[str(neighbour)] = orientation[str(face)] == coorintated

                            visited = True
                            break

                    if visited:
                        break

        return not (False in used.values())

    def Euler(self):
        return len(self.vertices) - len(self.edges) + len(self.faces)
