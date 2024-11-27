from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from mesa.batchrunner import batch_run

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128

import seaborn as sns

from collections import deque

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import json
import re

from random import choices




class PlayerAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.actionPoints = 4
        self.extraActionPoints = 0
        self.totalAPUsed = 0
        self.isCarryingVictim = False
        self.isKnockedOut = False
        self.victimsRescued = 0
        self.wasKnockedOut = 0
        self.cellsVisited = 0



# Funciones para leer el archvio con la descripción del tablero.

def load_scenario(file_path):
    """
    Procesa un archivo de texto con la configuración inicial del escenario y retorna los datos necesarios.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Leer la cuadrícula del escenario con las paredes
    walls = []
    for i in range(6):  # 6 filas de celdas
        row = lines[i].strip().split()
        walls.append([list(map(int, cell)) for cell in row])

    # walls = np.array(walls)  # Convertir a numpy array para facilitar operaciones

    # Leer los marcadores de puntos de interés
    markers = []
    for i in range(6, 9):  # 3 líneas de marcadores
        line = lines[i].strip().split()
        row, col, marker_type = int(line[0]), int(line[1]), line[2]
        markers.append((row, col, marker_type))  # Ajustar índices a 0-based

    # Leer los marcadores de fuego
    fires = []
    for i in range(9, 19):  # 10 líneas de marcadores de fuego
        line = lines[i].strip().split()
        row, col = int(line[0]), int(line[1])
        fires.append((row, col))

    # Leer las puertas
    doors = []
    for i in range(19, 27):  # 8 líneas de puertas
        line = lines[i].strip().split()
        r1, c1, r2, c2 = map(int, line)
        doors.append([[r1, c1], [r2, c2]])

    # Leer los puntos de entrada
    entry_points = []
    for i in range(27, 31):  # 4 líneas de puntos de entrada
        line = lines[i].strip().split()
        row, col = int(line[0]), int(line[1])
        entry_points.append((row, col))  # Ajustar índices a 0-based

    return walls, markers, fires, doors, entry_points



def get_grid(model):
  grid = np.zeros( (model.grid.height, model.grid.width) )

  for content, (x, y) in model.grid.coord_iter(): 
    grid[y][x] = model.board[y][x]["fireState"]
    if model.board[y][x]["fireState"] == 1:
      grid[y][x] = 2
    if model.board[y][x]["fireState"] == 2:
      grid[y][x] = 3
    if model.board[y][x]["marker"] == 1:
      grid[y][x] = 4
    if model.board[y][x]["marker"] == 2:
      grid[y][x] = 5

  for agent in model.schedule.agents:
    x,y = agent.pos
    grid[y][x] = 1

    
  return grid

def get_wall_grid(model):
    wall_grid = []
    for i in range(0, model.grid.height):
        row = []
        for j in range(0, model.grid.width):
            cell = model.board[i][j]["wall"].copy()
            row.append(cell.copy())
        wall_grid.append(row)
    return wall_grid

def get_damage_grid(model):
  damage_grid = []
  for i in range(0, model.grid.height):
    row = []
    for j in range(0, model.grid.width):
      cell = model.board[i][j]["damage"].copy()
      row.append(cell.copy())
    damage_grid.append(row)
  return damage_grid

def get_door_grid(model):
  door_grid = []
  for i in range(0, model.grid.height):
    row = []
    for j in range(0, model.grid.width):
      cell = model.board[i][j]["door"].copy()
      row.append(cell.copy())
    door_grid.append(row)
  return door_grid

def get_marker_grid(model):
  marker_grid = []
  for i in range(0, model.grid.height):
    row = []
    for j in range(0, model.grid.width):
      cell = model.board[i][j]["marker"]
      row.append(cell)
    marker_grid.append(row)
  return marker_grid

def get_revealed_grid(model):
  revealed_grid = []
  for i in range(0, model.grid.height):
    row = []
    for j in range(0, model.grid.width):
      cell = model.board[i][j]["revealed"]
      row.append(cell)
    revealed_grid.append(row)
  return revealed_grid

def get_fire_grid(model):
  fire_grid = []
  for i in range(0, model.grid.height):
    row = []
    for j in range(0, model.grid.width):
      cell = model.board[i][j]["fireState"]
      row.append(cell)
    fire_grid.append(row)
  return fire_grid


def get_agent_positions(model):
  agent_data = []
  for agent in model.schedule.agents:
    ida = agent.unique_id
    x,y = agent.pos
    agent_data.append([x, y, ida])
  return agent_data




def generate_board(width, height):
    board = []
    for i in range(0, height+2):
        row = []
        for j in range(0, width+2):
            cell = {"wall" : [0,0,0,0],"fireState" : 0, "marker" : 0, "revealed": 0, "damage" : [0,0,0,0], "door" : [0,0,0,0]}
            row.append(cell.copy())
        board.append(row)

    return board 


# Modelo para la simulación

class FireRescueModel(Model):
  def __init__(self, walls, markers, fires, doors, entry_points, players):
    super().__init__()
    self.victims = 0
    self.rescued = 0
    self.steps = 0
    self.ending = False
    self.false_markers = 5
    self.true_markers = 10
    self.grid = MultiGrid(10, 8, torus = False)
    self.schedule = BaseScheduler(self)
    self.datacollector = DataCollector( # Datos a recolectar para generar el reporte.
        model_reporters = {"Grid" : get_grid,
                           "WallGrid" : get_wall_grid,
                           "DamageGrid" : get_damage_grid,
                           "DoorsGrid" : get_door_grid,
                           "MarkerGrid" : get_marker_grid,
                           "RevealedGrid" : get_revealed_grid,
                           "FireGrid" : get_fire_grid,
                           "Steps" : lambda model: model.steps,
                           "Fire" : lambda model: sum(1 for i in range(0, 10) for j in range(0, 8) if model.board[j][i]["fireState"] == 2), # / model.fire_board.size,
                           "Damage" : lambda model: model.damage,
                           "Victims" : lambda model: model.victims,
                           "Rescued" : lambda model: model.rescued,
                           "GameEnd" : lambda model: model.ending,
                           "AgentInfo" : get_agent_positions},
                           
        # agent_reporters = {"Efficiency": lambda agent: agent.boxes_gathered / agent.energy_used,
        #                    "Resources_gathered": lambda agent: agent.boxes_gathered,
        #                    "Energy_used": lambda agent: agent.energy_used}
                           )


    width = 8
    height = 6
    self.board = generate_board(width, height) # Generar el tablero.
    self.damage = 0                                                                                
    
    # Generate initial fire
    for cord in fires:
      x,y = cord
      self.board[x][y]["fireState"] = 2
      
      # print("Fuegos en ",x,y,self.board[x][y])

    # Generate people to rescue
    for cord in markers:
      x,y,v = cord
      if v == "f":
        v = 1
      elif v == "v":
        v = 2
      self.board[x][y]["marker"] = v

    # Generate doors
    for pair in doors:
      p1, p2 = pair
      x1, y1 = p1
      x2, y2 = p2
          # Determinar la relación entre las celdas
      if x1 == x2 and y1 == y2 + 1:  # cell2 está a la izquierda de cell1
          self.board[x1][y1]["door"][1] = 1  # Puerta a la izquierda en cell1
          self.board[x2][y2]["door"][3] = 1  # Puerta a la derecha en cell2
      elif x1 == x2 and y1 == y2 - 1:  # cell2 está a la derecha de cell1
          self.board[x1][y1]["door"][3] = 1  # Puerta a la derecha en cell1
          self.board[x2][y2]["door"][1] = 1  # Puerta a la izquierda en cell2
      elif x1 == x2 + 1 and y1 == y2:  # cell2 está arriba de cell1
          self.board[x1][y1]["door"][0] = 1  # Puerta arriba en cell1
          self.board[x2][y2]["door"][2] = 1  # Puerta abajo en cell2
      elif x1 == x2 - 1 and y1 == y2:  # cell2 está abajo de cell1
          self.board[x1][y1]["door"][2] = 1  # Puerta abajo en cell1
          self.board[x2][y2]["door"][0] = 1  # Puerta arriba en cell2


    # print(self.board)
    for y in range(len(walls)):
      for x in range(len(walls[0])):
        self.board[y+1][x+1]["wall"] = walls[y][x]    

    # Break entrance walls
    print(entry_points)
    for pos in entry_points:
      y, x = pos
      if x - 1 == 0:
        self.board[y][x]["wall"][1] = 0
      if x + 1 == 9:
        self.board[y][x]["wall"][3] = 0
      if y - 1 == 0:
        self.board[y][x]["wall"][0] = 0
      if y + 1 == 7:
        self.board[y][x]["wall"][2] = 0
         
    
    print("Initial board: ", self.board)

    for i in range(players): # Generar a los agentes.
      agent = PlayerAgent(i, self)
      self.grid.place_agent(agent, (0,i))
      self.schedule.add(agent)
    
    self.datacollector.collect(self)

    

  def check_walls(self, pos):
    x, y = pos
    blockedPos = []
    blockedOrientations = []
    walls = self.board[y][x]["wall"]
    doors = self.board[y][x]["door"]
    if walls[0] == 1 and doors[0] != 2:
      blockedPos.append((x,y+1))
      blockedOrientations.append(0)
      
    if walls[1] == 1 and doors[1] != 2:
      blockedPos.append((x-1,y))
      blockedOrientations.append(1)

    if walls[2] == 1 and doors[2] != 2:
      blockedPos.append((x,y-1))
      blockedOrientations.append(2)

    if walls[3] == 1 and doors[3] != 2:
      blockedPos.append((x+1,y))
      blockedOrientations.append(3)

    return blockedPos, blockedOrientations

    
  def generate_random_fire(self):
    x = self.random.randrange(1,9)
    y = self.random.randrange(1,7)
    
    if self.board[y][x]["fireState"] == 0:
      self.board[y][x]["fireState"] = 1
    elif self.board[y][x]["fireState"] == 1:
      self.board[y][x]["fireState"] = 2

    elif self.board[y][x]["fireState"] == 2:
      self.spread_fire((x, y))
    

  def spread_fire2(self, pos, visited):
    x, y = pos
    adyacentCells = self.grid.get_neighborhood(pos, moore=False, include_center=False)
    walls, orientations = self.check_walls(pos)

    for cell in adyacentCells:
      x2, y2 = cell
      if cell not in walls:  

        if cell not in visited:
          visited.append(cell)

          if self.board[y2][x2]["fireState"] == 0:
            self.board[y2][x2]["fireState"] = 1

          elif self.board[y2][x2]["fireState"] == 1:
            self.board[y2][x2]["fireState"] = 2

          elif self.board[y2][x2]["fireState"] == 2:
            self.spread_fire(cell, visited)
            
      elif cell in walls:
          if cell not in visited:
            visited.append(cell)
            i = walls.index(cell)
            n = orientations[i]
            neighbor_offset = {
              0: (0, 1),  # arriba
              2: (0, -1),   # abajo
              3: (1, 0),   # derecha
              1: (-1, 0)  # izquierda
            }
            dx, dy = neighbor_offset[n]
            neighbor_x, neighbor_y = x + dx, y + dy


            if self.board[y][x]["door"][n] == 0:
              if self.board[y][x]["damage"][n] < 2:
                self.board[y][x]["damage"][n] += 1
                self.damage += 1
                print(self.steps, self.damage,  "damage in ", x, y)

              if self.board[y][x]["damage"][n] == 2:

                self.board[y][x]["wall"][n] = 0


              if 0 < neighbor_x < 10 and 0 < neighbor_y < 8:

                  opposite_orientation = {0: 2, 2: 0, 1: 3, 3: 1}[n]

                  if self.board[neighbor_y][neighbor_x]["damage"][opposite_orientation] < 2:
                      self.board[neighbor_y][neighbor_x]["damage"][opposite_orientation] += 1

                  if self.board[neighbor_y][neighbor_x]["damage"][opposite_orientation] == 2:
                      self.board[neighbor_y][neighbor_x]["wall"][opposite_orientation] = 0

            # Si la explosión es en una puerta
            elif self.board[y][x]["door"][n] == 2 or self.board[y][x]["door"][n] == 1:

              self.board[y][x]["door"][n] = 0
              self.board[y][x]["wall"][n] = 0


              if 0 < neighbor_x < 10 and 0 < neighbor_y < 8:

                opposite_orientation = {0: 2, 2: 0, 1: 3, 3: 1}[n]

                if self.board[neighbor_y][neighbor_x]["door"][opposite_orientation] == 2 or self.board[neighbor_y][neighbor_x]["door"][opposite_orientation] == 1:
                    self.board[neighbor_y][neighbor_x]["wall"][opposite_orientation] = 0
                    self.board[neighbor_y][neighbor_x]["door"][opposite_orientation] = 0



  def spread_fire(self, pos):
      queue = deque([pos])  # Cola para manejar las celdas pendientes por visitar
      visited = set()       # Conjunto para guardar las celdas ya visitadas
      damaged_walls = set() # Conjunto para rastrear paredes dañadas durante el turno

      while queue:
          x, y = queue.popleft()
          if (x, y) in visited:
              continue

          visited.add((x, y))
          adyacentCells = self.grid.get_neighborhood((x, y), moore=False, include_center=False)
          walls, orientations = self.check_walls((x, y))
          neighbor_offset = {
            0: (0, 1),  # arriba
            2: (0, -1),   # abajo
            3: (1, 0),   # derecha
            1: (-1, 0)  # izquierda
          }

          for cell in adyacentCells:
              x2, y2 = cell


              if cell in walls:
                  # Lógica para paredes y puertas
                  if (x2, y2) not in visited:
                      i = walls.index(cell)
                      n = orientations[i]
                      wall_id = ((x, y), n)  # Identificador único para la pared

                      # Manejo de la lógica de explosión para paredes y puertas
                      if wall_id not in damaged_walls:
                          if self.board[y][x]["door"][n] == 0:  # Pared
                              if self.board[y][x]["damage"][n] < 2:
                                  self.board[y][x]["damage"][n] += 1
                                  self.damage += 1
                                  damaged_walls.add(wall_id)  # Marcar la pared como dañada

                              if self.board[y][x]["damage"][n] == 2:
                                  self.board[y][x]["wall"][n] = 0  # Destruir pared

                              # Daño a la pared opuesta si existe
                              dx, dy = neighbor_offset[n]
                              neighbor_x, neighbor_y = x + dx, y + dy
                              if 0 <= neighbor_x < len(self.board[0]) and 0 <= neighbor_y < len(self.board):
                                  opposite_orientation = {0: 2, 2: 0, 1: 3, 3: 1}[n]
                                  opposite_wall_id = ((neighbor_x, neighbor_y), opposite_orientation)

                                  if opposite_wall_id not in damaged_walls:
                                      if self.board[neighbor_y][neighbor_x]["damage"][opposite_orientation] < 2:
                                          self.board[neighbor_y][neighbor_x]["damage"][opposite_orientation] += 1
                                          damaged_walls.add(opposite_wall_id)  # Marcar la pared como dañada

                                      if self.board[neighbor_y][neighbor_x]["damage"][opposite_orientation] == 2:
                                          self.board[neighbor_y][neighbor_x]["wall"][opposite_orientation] = 0

                          # Si la explosión es en una puerta
                          elif self.board[y][x]["door"][n] in [1, 2]:  # Puerta
                              self.board[y][x]["door"][n] = 0
                              self.board[y][x]["wall"][n] = 0
                              damaged_walls.add(wall_id)  # Marcar la pared como dañada

                              # Destruir puerta en la posición opuesta
                              dx, dy = neighbor_offset[n]
                              neighbor_x, neighbor_y = x + dx, y + dy
                              if 0 <= neighbor_x < len(self.board[0]) and 0 <= neighbor_y < len(self.board):
                                  opposite_orientation = {0: 2, 2: 0, 1: 3, 3: 1}[n]
                                  opposite_wall_id = ((neighbor_x, neighbor_y), opposite_orientation)

                                  if opposite_wall_id not in damaged_walls:
                                      self.board[neighbor_y][neighbor_x]["door"][opposite_orientation] = 0
                                      self.board[neighbor_y][neighbor_x]["wall"][opposite_orientation] = 0
                                      damaged_walls.add(opposite_wall_id)  # Marcar la pared como dañada
                  continue

              elif cell not in walls:  
                  # Expansión de fuego o humo
                  if self.board[y2][x2]["fireState"] == 0:
                      self.board[y2][x2]["fireState"] = 1  # Humo
                  elif self.board[y2][x2]["fireState"] == 1:
                      self.board[y2][x2]["fireState"] = 2  # Fuego
                  elif self.board[y2][x2]["fireState"] == 2:
                      queue.append(cell)  # Añadir a la cola para expandir el fuego



  def check_smoke(self):
    for j in range(0, self.grid.height):
        for i in range(0, self.grid.width):
            if self.board[j][i]["fireState"] == 1:
              neighbors = self.grid.get_neighborhood((i, j), moore=False, include_center=False)
              walls, o = self.check_walls((i, j))
              for cell in neighbors:
                if cell not in walls:
                  x, y = cell
                  if self.board[y][x]["fireState"] == 2:
                    self.board[j][i]["fireState"] = 2



  def check_fire(self):
    for j in range(0, self.grid.height):
        for i in range(0, self.grid.width):
            if self.board[j][i]["fireState"] == 2:
              if self.board[j][i]["marker"] == 2:
                self.board[j][i]["marker"] = 0
                self.victims += 1
              elif self.board[j][i]["marker"] == 1:
                self.board[j][i]["marker"] = 0

  def check_markers(self):
    totalMarkers = 0
    for j in range(0, self.grid.height):
        for i in range(0, self.grid.width):
            if self.board[j][i]["marker"] == 1  or self.board[j][i]["marker"] == 2:
              totalMarkers += 1
    markers = 3 - totalMarkers

    while markers > 0:
      chosen = choices(['true', 'false'], weights=[self.true_markers, self.false_markers], k=1)[0]

        # Reducir el número de marcadores según el tipo seleccionado
      x = self.random.randrange(1,9)
      y = self.random.randrange(1,7)
      if self.board[y][x]["fireState"] == 0 and self.board[y][x]["marker"] == 0:
        if chosen == 'true':
          self.true_markers -= 1
          self.board[y][x]["marker"] = 2
          markers -= 1

        else:
          self.false_markers -= 1
          self.board[y][x]["marker"] = 1
          markers -= 1
    
       
 

  def is_game_finished(self):
    if self.damage >= 24:
      return True
    if self.victims > 3:
      return True
    return False
  

  def step(self): 
    if not self.is_game_finished():
      self.steps += 1
      self.generate_random_fire()
      self.check_smoke()
      self.check_fire()
      self.check_markers()

    self.ending = self.is_game_finished()
    self.datacollector.collect(self)
    self.schedule.step()


walls, markers, fires, doors, entry_points = load_scenario("board.txt")

players = 6

model = FireRescueModel(walls, markers, fires, doors, entry_points, players)
Max_steps = 100

while not model.ending and Max_steps > 0:
  model.step()
  Max_steps -=1

print("Steps: ", model.steps)
print("Damage: ", model.damage)
print("Victims: ", model.victims)





all_grids = model.datacollector.get_model_vars_dataframe()
all_grids.head(10)




def convert_wall_grid_to_string(wall_grid):
    result = ""
    for i in range(1, len(wall_grid)-1):
        row_strings = []
        for j in range(1, len(wall_grid[0])-1):
            # Convertir la lista de 4 elementos en una cadena de bits
            wall_string = "".join(map(str, wall_grid[i][j]))
            row_strings.append(wall_string)
        # Unir las celdas de una fila con espacios y añadir la fila al resultado
        result += " ".join(row_strings) + "\n"
    
    return result.strip()

def convert_marker_pos_to_string(marker_grid, revealed_grid):
    result = ""
    pos_strings = []
    for i in range(1, len(marker_grid)-1):
        
        for j in range(1, len(marker_grid[0])-1):
            if marker_grid[i][j] == 1 and revealed_grid[i][j] == 0:
                pos_strings.append([i,j,"f","f"])
            elif marker_grid[i][j] == 2 and revealed_grid[i][j] == 0:
                pos_strings.append([i,j,"v","f"])

            if marker_grid[i][j] == 1 and revealed_grid[i][j] == 1:
                pos_strings.append([i,j,"f","v"])
            elif marker_grid[i][j] == 2 and revealed_grid[i][j] == 1:
                pos_strings.append([i,j,"v","v"])
        
    for string in pos_strings:
        result += f"{string[0]} {string[1]} {string[2]} {string[3]}\n"

    return result.strip()


def convert_fire_pos_to_string(fire_grid):
    result = ""
    total = 0
    pos_strings = []
    for i in range(1, len(fire_grid)):
        
        for j in range(1, len(fire_grid[0])):
            if fire_grid[i][j] == 1:
                pos_strings.append([i,j,"h"])
            elif fire_grid[i][j] == 2:
                pos_strings.append([i,j,"f"])

    for string in pos_strings:
        total += 1
        result += f"{string[0]} {string[1]} {string[2]} \n"

    result = f"{total} \n{result}"
    return result.strip()


def convert_door_pos_to_string(doors_grid):
    result = ""
    total = 0
    pos_strings = []
    for i in range(1, len(doors_grid)-1):
        for j in range(1, len(doors_grid[0])-1):
            for k in range(0,3):
                
                if doors_grid[i][j][k] == 1:
                    if k == 0:
                        pos_strings.append([i-1,j, i,j,0])
                    elif k == 1:
                        pos_strings.append([i,j-1, i,j,0])

                elif doors_grid[i][j][k] == 2:
                    if k == 0:
                        pos_strings.append([i-1,j, i,j,1])
                    elif k == 1:
                        pos_strings.append([i,j-1, i,j,1])


    for string in pos_strings:
        total += 1
        result += f"{string[0]} {string[1]} {string[2]} {string[3]} {string[4]} \n"

    result = f"{total} \n{result}"
    return result.strip()


def convert_entry_points_pos_to_string(entry_points):
    result = ""
    total = 0
    for cord in entry_points:
        total += 1
        result += f"{cord[0]} {cord[1]}\n"

    result = f"{total} \n{result}"
    return result.strip()

def convert_agent_data_to_string(agent_data):
    result = ""
    total = 0
    for data in agent_data:
        total += 1
        result += f"{data[0]} {data[1]} {data[2]} \n"
    result = f"{total} \n{result}"
    return result.strip()



def get_end(step):
    print(all_grids["GameEnd"][step])
    if all_grids["GameEnd"][step]:
        return "true"
    return "false"

def get_json(step):
    wall_grid_string = convert_wall_grid_to_string(all_grids["WallGrid"][step])

    markers_string =convert_marker_pos_to_string(all_grids["MarkerGrid"][step], all_grids["RevealedGrid"][step])

    fire_string = convert_fire_pos_to_string(all_grids["FireGrid"][step])

    door_string = convert_door_pos_to_string(all_grids["DoorsGrid"][step])

    entry_points_string = convert_entry_points_pos_to_string(entry_points)

    agent_data_string = convert_agent_data_to_string(all_grids["AgentInfo"][0])

    jsonString = f"{wall_grid_string} \n{markers_string} \n{fire_string} \n{door_string} \n{entry_points_string} \n{agent_data_string}"

    return jsonString

print(get_json(0))






responseStep = 0

class Server(BaseHTTPRequestHandler):

    def do_GET(self):
        global responseStep

        path = self.path
        print("path: ", path)
        match = re.match(r'^/(\d+)$', path)
        print("match: ", match)
        if match:
            responseStep = int(match.group(1))
            print("response step: ", responseStep)

        else:
            # Si no se proporciona un número válido
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            responseStep += 1
            end = get_end(responseStep)
            content = get_json(responseStep)
            # Crear la respuesta en formato JSON
            response = {
                "step" : responseStep,
                "content" : content,
                "end" : end
            }

            response_json = json.dumps(response)

            self.wfile.write(response_json.encode('utf-8'))
            return

        # self._set_response()
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        end = get_end(responseStep)
        content = get_json(responseStep)
        # Crear la respuesta en formato JSON
        response = {
            "step" : responseStep,
            "content" : content,
            "end" : end
        }

        response_json = json.dumps(response)


        self.wfile.write(response_json.encode('utf-8'))



def run(server_class=HTTPServer, handler_class=Server, port=8585):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info(f"Starting httpd in port {port}...\n") # HTTPD is HTTP Daemon!
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:   # CTRL+C stops the server
        pass
    httpd.server_close()
    logging.info("Stopping httpd...\n")

run()