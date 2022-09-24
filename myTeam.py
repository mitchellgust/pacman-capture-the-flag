# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from array import array
import random
from capture import GameState
from captureAgents import CaptureAgent
from game import Actions, Directions
import util

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent', numTraining=0):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

# Common Commands:
# 
# Run against baseline team:
# python capture.py -r myTeam -b baselineTeam
# 
# Run against itself:
# python capture.py -r myTeam -b myTeam
# 
# Run with different layout:
# 
# python capture.py -r myTeam -b baselineTeam -l officeCapture
# ---------------
class Node:
  def __init__(self, state, path, cost):
    self.state = state
    self.path = path
    self.cost: int  = cost

  @property
  def getState(self):
    return self.state

  @property
  def getPath(self):
    return self.path

  @property
  def getCost(self):
    return self.cost

class ValueMap:
  rows: int
  columns: int
  valueMap: list

  def __init__(self, rows, columns):
    self.rows = rows
    self.columns = columns
    self.valueMap = self.initialise2DArray(rows, columns)

  def initialise2DArray(self, rows, columns):
    # initialise 2d array
    valueMap = []
    for row in range(rows):
      valueMap.append([])
      for column in range(columns):
        valueMap[row].append(" ")

    return valueMap
  
  def translateCoordinate(self, row, column, toMap):
    if toMap == "pacman":
      return (column, self.rows - 1 - row)
    else:
      return (self.rows - 1- column, row)


    
  def printValueMap(self):
    for i in range(self.rows):
      for j in range(self.columns):
        if self.valueMap[i][j] is None:
          print("WA", end=" ")
        else:
          print(self.valueMap[i][j], end=" ")
      print()

  def getNorth(self, row, column):
    return self.valueMap[row - 1][column]
  
  def getSouth(self, row, column):
    return self.valueMap[row + 1][column]

  def getEast(self, row, column):
    return self.valueMap[row][column + 1]

  def getWest(self, row, column):
    return self.valueMap[row][column - 1]

  @property
  def getRows(self):
    return self.rows

  @property
  def getColumns(self):
    return self.columns

  @property
  def getValueMap(self):
    return self.valueMap
  
  def __getitem__(self, key):
    return self.valueMap[key]
  
  def __setitem__(self, key, value):
    self.valueMap[key] = value

RED_TEAM_OFFSET = -1
BLUE_TEAM_OFFSET = 0
ALL_ACTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
STOP_ACTION = Directions.STOP

class BaselineAgent(CaptureAgent):
  def registerInitialState(self, gameState: GameState):
    # Common Variables that are Constant
    self.wallPositions = gameState.getWalls().asList()

    self.mapWidth = gameState.data.layout.width
    self.mapHeight = gameState.data.layout.height
    midWidth = int(self.mapWidth / 2)
    midHeight = int(self.mapHeight / 2)
    
    # Calculate Middle of Map Depending on Team
    self.isRed = self.index in gameState.getRedTeamIndices()
    offset = RED_TEAM_OFFSET if self.isRed else BLUE_TEAM_OFFSET
    self.middleOfMap = (midWidth + offset, midHeight)

    # Calculate Entrance Positions
    entrancePositions = []
    for i in range(self.mapHeight):
      if not gameState.hasWall(midWidth + offset, i) and not gameState.hasWall(midWidth, i):
        entrancePositions.append((midWidth + offset, i))
    self.entrancePositions = entrancePositions
    


  def getSuccessors(self, state, walls):
      successors = []
      for action in ALL_ACTIONS: # Includes Stop Action
        x, y = state
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        if (nextx, nexty) not in walls:
          new_node = Node((nextx, nexty), action, 1)
          successors.append(new_node)
      return successors
      
  def aStarSearch(self, start_position, goal_position, walls, heuristic):
    def priorityQueueFunction(node: Node): 
      # f(n) = g(n) + h(n) (priority = cost + estimatedCost)
      return node.cost + heuristic(node.state, goal_position)

    priorityQueue = util.PriorityQueueWithFunction(priorityQueueFunction)
    initialNode = Node(start_position, [], 0)
    frontier = priorityQueue
    frontier.push(initialNode)

    reached = []
    while not frontier.isEmpty():
      node: Node = frontier.pop()
      state = node.state
      path = node.path
      cost = node.cost

      if state == goal_position:
        if not path:
          return STOP_ACTION
        else:
          return path[0] 
   
      if state in reached:
          continue
      
      reached.append(state)
      
      successor: Node
      for successor in self.getSuccessors(state, walls):
          successorPath = path + [successor.path]
          successorCost = cost + successor.cost
          successorNode = Node(successor.state, successorPath, successorCost)
          
          frontier.push(successorNode)

    return None # Failure


class OffensiveReflexAgent(BaselineAgent):
  def registerInitialState(self, gameState):
    self.lastPositionReward = -1
    self.scaredGhostReward = 20
    self.foodReward = 30
    self.ghostReward = -1000
    self.capsuleReward = 40
    self.deadendReward = -400
    self.emptyLocationReward = -0.1
    self.gamma = 0.9


    self.enemyClose = False
    self.gamma = 1
    super().registerInitialState(gameState)
    CaptureAgent.registerInitialState(self, gameState)
    self.holdingPoints = 0
    # Get CONSTANT variables
    self.currentPosition = gameState.getAgentPosition(self.index)
    self.initialPosition = gameState.getAgentPosition(self.index)
    self.capsulePositions = self.getCapsulesYouAreOffending(gameState)

    # Get INITIAL Food Positions
    self.foodPositions = self.getFoodYouAreOffending(gameState).asList()

    openPositions = []
    # get all positions that have just 1 wall around them
    for i in range(self.mapWidth):
      for j in range(self.mapHeight):
        # check if position is a wall
        if gameState.hasWall(i, j):
          continue
        # check if position is adjacent to the edge of the map
        if i == 0 or i == self.mapWidth - 1 or j == 0 or j == self.mapHeight - 1:
          continue
        # check if position has 3 walls around it
        wallCount = 0
        xValues = [i - 1, i + 1, i, i]
        yValues = [j, j, j - 1, j + 1]
        for x, y in zip(xValues, yValues):
          if gameState.hasWall(x, y):
            wallCount += 1
        if wallCount < 2:
          openPositions.append((i, j))

    positionDistToOpenPositionMap = {}
    for i in range(self.mapWidth):
      for j in range(self.mapHeight):
        if gameState.hasWall(i, j):
          continue
        distancesToOpenPositions = [self.getMazeDistance(
            (i,j), openPosition) for openPosition in openPositions]
        minDistance = min(distancesToOpenPositions)
        positionDistToOpenPositionMap[(
            i, j)] = minDistance

    self.positionDistToOpenPositionMap = positionDistToOpenPositionMap



    # Get INITIAL Enemy Positions
    self.enemyPositions = []
    self.scaredEnemyPositions = []
    self.enemies = self.getEnemiesYouAreOffending(gameState)


    self.valueMap = ValueMap(self.mapHeight, self.mapWidth)

    for row in range(self.valueMap.rows):
      for column in range(self.valueMap.columns):
        self.valueMap[row][column] = OffensiveReflexAgent.rewardFunction(self, 
          self.valueMap.translateCoordinate(row, column, "pacman"))



  def getFoodYouAreOffending(self, gameState):
    if self.isRed:
      return gameState.getBlueFood()
    else:
      return gameState.getRedFood()

  def getCapsulesYouAreOffending(self, gameState):
    if self.isRed:
      return gameState.getBlueCapsules()
    else:
      return gameState.getRedCapsules()
  
  def getEnemiesYouAreOffending(self, gameState):
    if self.isRed:
      return gameState.getBlueTeamIndices()
    else:
      return gameState.getRedTeamIndices()

  def chooseAction(self, gameState : GameState):
    # filter positionDistToOpenPositionMap by value greater than 5
    # claustrophobicPositions ={
    #     k: v for (k, v) in self.positionDistToOpenPositionMap.items() if v > 0}
    # self.debugDraw(list(claustrophobicPositions.keys()), [0, 1, 0], clear=True)

    self.currentPosition = gameState.getAgentPosition(self.index)
    
    # Get NEW Current Food Positions
    self.foodPositions = self.getFoodYouAreOffending(gameState).asList()
    self.capsulePositions = self.getCapsulesYouAreOffending(gameState)

    gameState = self.getCurrentObservation()
    enemyIndexes = self.getOpponents(gameState)
    observableEnemyIndexes = [
        enemyIndex for enemyIndex in enemyIndexes if gameState.getAgentPosition(enemyIndex)]
    self.scaredEnemyPositions = [gameState.getAgentPosition(
        enemyIndex) for enemyIndex in observableEnemyIndexes if gameState.getAgentState(enemyIndex).scaredTimer > 0]
    self.enemyPositions = [gameState.getAgentPosition(
        enemyIndex) for enemyIndex in observableEnemyIndexes if gameState.getAgentState(enemyIndex).scaredTimer == 0]
    
    self.enemyClose = False
    for enemyPos in self.enemyPositions:
      if self.getMazeDistance(self.currentPosition, enemyPos) < 5:
        self.enemyClose = True


    if self.getPreviousObservation():
      previousFoods = self.getPreviousObservation().getBlueFood().asList(
      ) if self.red else self.getPreviousObservation().getRedFood().asList()
      missingFoods = [
          food for food in previousFoods if food not in self.getFood(gameState).asList()]
      if len(missingFoods) > 0:
        self.holdingPoints += 1
      if self.currentPosition in self.entrancePositions:
        self.holdingPoints = 0
      if self.holdingPoints > 2:
        closestEntrance = min(
            self.entrancePositions, key=lambda x: self.getMazeDistance(self.currentPosition, x))
        best_action = self.aStarSearch(
            self.currentPosition, closestEntrance, self.wallPositions, util.manhattanDistance)
        return best_action

    self.valueIteration()
    
    # which action is best
    rowPac, colPac =  self.currentPosition
    valueMapPos = self.valueMap.translateCoordinate(rowPac, colPac, "value")
    row = valueMapPos[0]
    col = valueMapPos[1]

    self.legalActions = gameState.getLegalActions(self.index)

    # Update Possible Actions
    actionValues = {}
    for action in self.legalActions:
      if action is Directions.NORTH:
        actionValues.update({action : self.valueMap.getNorth(row, col)})
      elif action is Directions.SOUTH:
        actionValues.update({action : self.valueMap.getSouth(row, col)})
      elif action is Directions.EAST:
        actionValues.update({action : self.valueMap.getEast(row, col)})
      elif action is Directions.WEST:
        actionValues.update({action : self.valueMap.getWest(row, col)})

    actionToTake = max(actionValues, key=actionValues.get)

    return actionToTake

  def rewardFunction(self, position):

    if position in self.wallPositions:
      return None
    else:
      reward = self.emptyLocationReward
      distToSelf = self.getMazeDistance(position, self.currentPosition)
      try:
        previousObservation = self.getPreviousObservation()
      except:
        previousObservation = None

      if position in self.enemyPositions:
        reward = self.ghostReward
        if distToSelf < 5 and distToSelf > 0:
          reward *= 5/distToSelf  
      elif position in self.foodPositions:
        if self.enemyClose:
          reward = -self.foodReward
        else:
          reward = self.foodReward
          if distToSelf <= 2 and distToSelf > 0:
            reward *= (2/distToSelf)
      elif position in self.capsulePositions:
        reward += self.capsuleReward
        if distToSelf <= 2 and distToSelf > 0:
          reward *= (2/distToSelf)
      elif position in self.scaredEnemyPositions:
        reward = self.scaredGhostReward
        if distToSelf <= 2 and distToSelf > 0:
          reward *= (2/distToSelf)

      if self.enemyClose:
        deadEndVal = self.positionDistToOpenPositionMap[position]
        if deadEndVal > 0 and distToSelf < 2:
          reward = self.deadendReward
      if position in [self.getPreviousObservation().getAgentPosition(self.index) if previousObservation is not None else None]:
        reward = self.lastPositionReward
      return reward

  def bellman(self, map: ValueMap, position):
    row, column = position
    up, down, left, right = None, None, None, None
  
    reward = self.rewardFunction(map.translateCoordinate(row, column, "pacman"))

    # If reward is none, it is a wall
    if reward is None:
      return None

    # up
    if row < self.mapHeight - 1:
      up = map.getNorth(row, column)
    # down
    if row > 0:
      down = map.getSouth(row, column)
    # right
    if column < self.mapWidth - 1:
      right = map.getEast(row, column)
    # left
    if column > 0:
      left = map.getWest(row, column)

    if up is None:
      up = -1
    if down is None:
      down = -1
    if right is None:
      right = -1
    if left is None:
      left = -1

    upValue = up * 0.8 + (right + left) * 0.1
    downValue = down * 0.8 + (right + left) * 0.1
    rightValue = right * 0.8 + (up + down) * 0.1
    leftValue = left * 0.8 + (up + down) * 0.1

    maxAction = max(upValue, downValue, rightValue, leftValue)
    return float(reward) + (self.gamma * float(maxAction))


  def valueIteration(self):
    iteration = 100
    while(iteration > 0):
      newMap: ValueMap = ValueMap(self.mapHeight, self.mapWidth)

      for row in range(self.mapHeight):
        for col in range(self.mapWidth):
          newMap[row][col] = self.bellman(self.valueMap, (row, col))
      
      iteration -= 1
      self.valueMap = newMap

class DefensiveReflexAgent(BaselineAgent):
  def registerInitialState(self, gameState: GameState):
    super().registerInitialState(gameState)
    
    self.currentPosition = gameState.getAgentPosition(self.index)
    self.currentTarget = None
    
    # TODO: Initialize variables we need here:
    # positions to food, enemy pacman, capsules and the entrances of 
    # the ally's side of the maze
    
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState: GameState):
    # TODO: Get information about the gameState
    # Is the agent scared? Should we stay away from the enemy but as close as possible?
    # Are there any enemies within 5 steps of the agent? Chase them!
    # Is there any food that was present in the last observation that 
    # was eaten? Go to that location.
    # 
    # Once we have decided what to do, we can call aStarSearch to find the best action
    
    # debug draw the entrances
    # self.debugDraw(self.entrancePositions, [0, 1, 0], clear=True)
    
    # # Information about the gameState and current agent
    currentPosition = gameState.getAgentPosition(self.index)
    lastState = self.getPreviousObservation()
    goalPosition = self.currentTarget if self.currentTarget else self.currentPosition
    isInvestigatingFood = False
    isChasingEnemy = False
    
    
    # Function 1
    # By default - Put Agent near the middle of the maze, priorititising the location to be in a conjestion of food
    # TODO: UNCOMMENT THIS TO GET EXAMPLE OF PACMAN EATING A CAPSULE
    if currentPosition == goalPosition:
      goalPosition = self.middleOfMap 
    
    # Function 2
    # If food is detected as eaten from the last observation, go to that location
    if lastState: 
      lastStateFoods = self.getFoodYouAreDefending(lastState).asList()
      currentStateFoods = self.getFoodYouAreDefending(gameState).asList()
      foodsEatenSinceLastState = [food for food in lastStateFoods if food not in currentStateFoods]
      
      if foodsEatenSinceLastState:
        isInvestigatingFood = True
        goalPosition = min(foodsEatenSinceLastState,
                            key=lambda x: self.getMazeDistance(currentPosition, x))
        
    # Function 3
    # If enemy is within observable range, chase them
    enemyIndexes = self.getOpponents(gameState)
    observableEnemyPositions = [
        gameState.getAgentPosition(enemyIndex) for enemyIndex in enemyIndexes if gameState.getAgentPosition(enemyIndex)]
    if observableEnemyPositions:
      closest_enemy = min(observableEnemyPositions,
                          key=lambda x: self.getMazeDistance(currentPosition, x))
      
      # so we don't chase into enemy territory
      if self.isRed:
        if closest_enemy[0] < self.middleOfMap[0]:
          goalPosition = closest_enemy
          isChasingEnemy = True
      else:
        if closest_enemy[0] > self.middleOfMap[0]:
          goalPosition = closest_enemy
          isChasingEnemy = True

    # Function 4
    # When Scared - Move away from the enemy pacman but stay relatively close
    if gameState.getAgentState(self.index).scaredTimer > 0:
      if isChasingEnemy:
        closestEnemyPosition = goalPosition
        firstSuccessorsOfEnemy = self.getSuccessors(
            closestEnemyPosition, self.wallPositions)
        secondSuccessorsOfEnemy = [successor for first_successor in firstSuccessorsOfEnemy for successor in self.getSuccessors(first_successor.state, self.wallPositions)]
        thirdSuccessorsOfEnemy = [successor for second_successor in secondSuccessorsOfEnemy for successor in self.getSuccessors(
            second_successor.state, self.wallPositions)]
        successors3AwayFromEnemy = [successor.state for successor in thirdSuccessorsOfEnemy if self.getMazeDistance(successor.state, closestEnemyPosition) == 3]
        # self.debugDraw(successors3AwayFromEnemy, [1, 0, 0], clear=False)
        successorClosestToCurrentPosition = min(successors3AwayFromEnemy, key=lambda x: self.getMazeDistance(currentPosition, x), default=self.middleOfMap)
        goalPosition = successorClosestToCurrentPosition
        
  
    best_action = self.aStarSearch(
        currentPosition, goalPosition, self.wallPositions, util.manhattanDistance)
    self.currentTarget = goalPosition   
    
    return best_action