def try_int(s, base=10, val=None):
  try:
    return int(s, base)
  except ValueError:
    return val

class Obj(object):
    def __init__(self, filename):
        with open(filename) as f:
            self.lines = f.read().splitlines() #separa el string y lo separa linea por linea

        self.vertices = []
        self.vfaces = []
        self.read()

    def read(self):
        for line in self.lines:
            if line:#si la linea no esta en blanco line.split ('',1) separar las lineas en dos, la f y el codigo
                prefix, value = line.split(' ', 1)  #prefix es la letra y value los numeros

                if prefix == 'v': #si encontre una v meto un nuevo valor
                    self.vertices.append(list(map(float, value.split(' '))))  #split de las cordenadas, asi salen separadas, y el float es porque se haran operaciones con el numero
                elif prefix == 'f':
                    self.vfaces.append([list(map(int , face.split('/'))) for face in value.split(' ')])  #para separar todos los / en un array y pasar a int tipo [2, 3, 4]