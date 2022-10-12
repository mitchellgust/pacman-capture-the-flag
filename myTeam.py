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
from cmath import inf
import random
from capture import GameState
from captureAgents import CaptureAgent
from game import Actions, Directions, Game
import util

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgentV2', second='DefensiveReflexAgent', numTraining=0):
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


RED_TEAM_OFFSET = -1
BLUE_TEAM_OFFSET = 0
ALL_ACTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]


class BaselineAgent(CaptureAgent):
  def registerInitialState(self, gameState: GameState):
    # Common Variables that are Constant
    self.wallPositions = gameState.getWalls().asList()

    self.mapWidth = gameState.data.layout.width
    self.mapHeight = gameState.data.layout.height
    middleColumn = int(self.mapWidth / 2)
    middleRow = int(self.mapHeight / 2)
    
    # Calculate Middle of Map Depending on Team
    self.isRed = self.index in gameState.getRedTeamIndices()
    offset = RED_TEAM_OFFSET if self.isRed else BLUE_TEAM_OFFSET
    self.middleOfMap = (middleColumn + offset, middleRow)

    # Calculate Entrance Positions
    self.entrancePositions = self.getEntrancePositions(gameState, middleColumn, offset)

  def getEntrancePositions(self, gameState: GameState, midWidth: int, teamOffset: int):
    entrancePositions = []

    # Get Entrance Positions
    for row in range(self.mapHeight):
      # If Space Empty on Left and Right of The Middle of the Map
      if not gameState.hasWall(midWidth - 1, row) and not gameState.hasWall(midWidth, row):
        entrancePositions.append((midWidth + teamOffset, row))
        
    return entrancePositions

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
          return Directions.STOP
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

  def isRowOrColumnEdge(self, row, column):
    return row == 0 or row == self.mapWidth - 1 or column == 0 or column == self.mapHeight - 1

  def getOpenPositions(self, gameState: GameState):
    self.openPositions = []
    for i in range(self.mapWidth):
      for j in range(self.mapHeight):
        # Don't Check Position that is a Wall
        if gameState.hasWall(i, j):
          continue

        # TODO: CHECK THIS APPLIES TO ALL GAMES
        if self.isRed:
          if i < self.middleOfMap[0]:
            continue 
        else:
          if i > self.middleOfMap[0]:
            continue

        # Don't Check Row or Column that is on Edge of Map
        if self.isRowOrColumnEdge(i, j):
          continue

        # For All Other Positions, Count Surrounding Walls
        wallCount = 0
        positionsToCheck = [(i, j - 1), (i + 1, j), (i, j + 1), (i - 1, j)] # North, East, South, West

        for x, y in positionsToCheck:
          if gameState.hasWall(x, y):
            wallCount += 1
        
        # If Position is Surrounded by 1 or 0 Walls, Add to List
        if wallCount < 2:
          self.openPositions.append((i, j))

    return self.openPositions

  def getDistanceMapToOpenPositions(self, gameState: GameState, openPositions):
    distanceMapToOpenPositions = {}
    # Get Distance from Every Position to Closest Open Position
    for i in range(self.mapWidth):
      for j in range(self.mapHeight):
        
        # Don't Check Position that is a Wall
        if gameState.hasWall(i, j):
          continue

        # Get Minimum Distance from Position to Every Open Position
        distancesToOpenPositions = []
        for openPosition in openPositions:
          distance = self.getMazeDistance((i, j), openPosition)
          distancesToOpenPositions.append(distance)
        
        minDistance = min(distancesToOpenPositions)

        # Add to Distance Map
        distanceMapToOpenPositions[(i, j)] = minDistance

    return distanceMapToOpenPositions


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
    # Is the agent scared? Should we stay away from the enemy but as close as possible?
    # Are there any enemies within 5 steps of the agent? Chase them!
    # Is there any food that was present in the last observation that 
    # was eaten? Go to that location.
    # 
    # Once we have decided what to do, we can call aStarSearch to find the best action

    # Information about the gameState and current agent
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
  

class OffensiveAgentV2(BaselineAgent):
  def registerInitialState(self, gameState: GameState):
    super().registerInitialState(gameState)
    # Constants
    self.mdpIterations = 20
    self.lastPositionReward = -1
    self.scaredGhostReward = 20
    self.foodReward = 30
    self.ghostReward = -1000
    self.capsuleReward = 40
    self.defaultReward = -0.1
    self.returnHomeThreshold = 2
    self.gamma = 0.9

    self.walls = gameState.getWalls().asList()
    self.mapWidth = gameState.data.layout.width
    self.mapHeight = gameState.data.layout.height
    self.scoreMap = self.getNewMap(self.walls)
    self.holdingPoints = 0
    
    self.distanceMapToOpenPositions = None
    
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState: GameState):
    # Create Map of Minimum Distances to Positions with 1 or 0 surrounding walls
    if self.distanceMapToOpenPositions is None:
      openPositions = self.getOpenPositions(gameState)
      self.distanceMapToOpenPositions = self.getDistanceMapToOpenPositions(gameState, openPositions)

    currentPosition = gameState.getAgentPosition(self.index)

    # Analyse New Move
    if self.getPreviousObservation():

      # Get Previous Food
      previousFood = []
      if self.red:
        previousFood = self.getPreviousObservation().getBlueFood().asList()
      else:
        previousFood = self.getPreviousObservation().getRedFood().asList()
      
      # If Food was Eaten in New Turn
      currentFood = self.getFood(gameState).asList()
      missingFood = []
      for food in previousFood:
        if food not in currentFood:
          missingFood.append(food)
      
      # Food was Eaten - Add to Holding Points
      if len(missingFood) > 0:
        self.holdingPoints += 1
      
      # Food is Returned - Reset Holding Points
      if currentPosition in self.entrancePositions:
        self.holdingPoints = 0      
      
      # Return Home is Threshold is Reached
      if self.holdingPoints > self.returnHomeThreshold:

        enemyIndexes = self.getOpponents(gameState)

        # Get Obvervable Enemies that are not Scared
        observableEnemyPositions = []
        for enemyIndex in enemyIndexes:
          enemyPosition = gameState.getAgentPosition(enemyIndex)
          if enemyPosition and gameState.getAgentState(enemyIndex).scaredTimer == 0:
            observableEnemyPositions.append(enemyPosition)

        # Get Entrances where Non Scared Enemy is at least 3 steps away
        if observableEnemyPositions:
          safeEntrances = []
          for entrance in self.entrancePositions:
            for enemyPosition in observableEnemyPositions:
              if self.getMazeDistance(entrance, enemyPosition) > 3:
                safeEntrances.append(entrance)
        else:
          # If No Observable Enemies, All Entrances are Safe
          safeEntrances = self.entrancePositions

        # Closest Safe Entrance
        closestEntrance = min(safeEntrances, key=lambda x: self.getMazeDistance(currentPosition, x))

        best_action = self.aStarSearch(
            currentPosition, closestEntrance, self.walls, util.manhattanDistance)
            
        return best_action

    # If not securing points, use MDP to update the score map
    self.scoreMap = self.valueIteration(self.scoreMap, gameState)
    legalActions = gameState.getLegalActions(self.index)

    return self.getBestActionFromScoreMap(legalActions, self.scoreMap, currentPosition)

  def getRewardMap(self, gameState: GameState):
    food = self.getFood(gameState).asList()
    walls = self.walls
    capsules = self.getCapsules(gameState)
    scoreMap = self.getNewMap(walls)
    width = self.mapWidth
    height = self.mapHeight

    gameState = self.getCurrentObservation()
    currentPosition = gameState.getAgentPosition(self.index)
    
    previousObservation = self.getPreviousObservation()
    previousPosition = previousObservation.getAgentPosition(self.index) if previousObservation else None
    
    enemyIndexes = self.getOpponents(gameState)
    observableEnemyIndexes = [
        enemyIndex for enemyIndex in enemyIndexes if gameState.getAgentPosition(enemyIndex)]
    scaredEnemyPositions = [gameState.getAgentPosition(
        enemyIndex) for enemyIndex in observableEnemyIndexes if gameState.getAgentState(enemyIndex).scaredTimer > 0]
    unscaredEnemyPositions = [gameState.getAgentPosition(
        enemyIndex) for enemyIndex in observableEnemyIndexes if gameState.getAgentState(enemyIndex).scaredTimer == 0]
    
    observableEnemyIndexesOfCurrentAgent = [
        enemyIndex for enemyIndex in observableEnemyIndexes if util.manhattanDistance(gameState.getAgentPosition(enemyIndex), currentPosition) <= 5]
    unscaredEnemyPositionsOfCurrentAgent = [gameState.getAgentPosition(
        enemyIndex) for enemyIndex in observableEnemyIndexesOfCurrentAgent if gameState.getAgentState(enemyIndex).scaredTimer == 0]
      
    
    positionsAdjacentToUnscaredEnemies = set()
    for enemyPosition in unscaredEnemyPositions:
      successors = self.getSuccessors(enemyPosition, walls)
      successorsAsTuples = [successor.state for successor in successors]
      positionsAdjacentToUnscaredEnemies.update(successorsAsTuples)
    positionsAdjacentToUnscaredEnemies = list(positionsAdjacentToUnscaredEnemies)
    
    for x in range(width):
      for y in range(height):
        cell = (x, y)
        if cell in walls:
          scoreMap[x][y] = None
        elif cell in unscaredEnemyPositions or cell in positionsAdjacentToUnscaredEnemies:
          scoreMap[x][y] = self.ghostReward
        elif cell in food:
          distanceToFood = self.getMazeDistance(currentPosition, cell)
          currentfoodReward = self.foodReward * 0.9 ** distanceToFood
          if len(unscaredEnemyPositionsOfCurrentAgent) > 0:
            foodRisk = self.distanceMapToOpenPositions[cell]
            scoreMap[x][y] = -200 if foodRisk > 1 else currentfoodReward
          else: 
            scoreMap[x][y] = currentfoodReward
        elif cell in scaredEnemyPositions:
          scoreMap[x][y] = self.scaredGhostReward
        elif cell in capsules:
          scoreMap[x][y] = self.capsuleReward
        elif cell in [previousPosition if previousObservation is not None else None]:
          scoreMap[x][y] = self.lastPositionReward
        else:
          scoreMap[x][y] = self.defaultReward
    return scoreMap

  def getNewMap(self, walls):
    width = self.mapWidth
    height = self.mapHeight

    scoreMap = []
    for x in range(width):
        scoreMap.append([])
        for y in range(height):
            scoreMap[x].append("  ")

    for x in range(width):
        for y in range(height):
            if (x, y) in walls:
                scoreMap[x][y] = None
            else:
                scoreMap[x][y] = self.defaultReward
    return scoreMap

  def getBestActionFromScoreMap(self, legalActions, scoreMap, position):
    x = position[0]
    y = position[1]
    actions = []
    actionScores = []
    
    for action in legalActions:
      if action is Directions.NORTH:
        value = scoreMap[x][y+1]
      elif action is Directions.SOUTH:
        value = scoreMap[x][y-1]
      elif action is Directions.EAST:
        value = scoreMap[x+1][y]
      elif action is Directions.WEST:
        value = scoreMap[x-1][y]
      actions.append(action)
      actionScores.append(value)
      
    # Try to get unstuck if unsure of what to do
    if len(actions) == 0:
      return random.choice(legalActions)

    maxScoreIdx = actionScores.index(max(actionScores))
    maxScoreChoice = actions[maxScoreIdx]
    return maxScoreChoice

  def bellmannUpdate(self, scoreMap, cell, reward):
    width = self.mapWidth
    height = self.mapHeight
    x = cell[0]
    y = cell[1]

    # If reward is none, it is a wall
    if reward is None:
      return None

    # left
    if x < width - 1:
      left = scoreMap[x + 1][y] if scoreMap[x + 1][y] is not None else -1
    # right
    if x > 0:
      right = scoreMap[x - 1][y] if scoreMap[x - 1][y] is not None else -1
    # up
    if y < height - 1:
      up = scoreMap[x][y + 1] if scoreMap[x][y + 1] is not None else -1
    # down
    if y > 0:
      down = scoreMap[x][y - 1] if scoreMap[x][y - 1] is not None else -1

    probability = 0.8
    probabilityOther = 0.1
    upValue = up * probability + (right + left) * probabilityOther
    downValue = down * probability + (right + left) * probabilityOther
    rightValue = right * probability + (up + down) * probabilityOther
    leftValue = left * probability + (up + down) * probabilityOther

    maxValue = max([upValue, downValue, rightValue, leftValue])
    return float(reward + self.gamma * maxValue)

  def valueIteration(self, scoreMap, gameState: GameState):
    iterations = self.mdpIterations
    walls = self.walls
    currentRewardMap = self.getRewardMap(gameState)
    width = self.mapWidth
    height = self.mapHeight

    while iterations > 0:
        newMap = self.getNewMap(walls)
        for x in range(width):
            for y in range(height):
                reward = currentRewardMap[x][y]
                newMap[x][y] = self.bellmannUpdate(scoreMap, (x, y), reward)
        scoreMap = newMap
        iterations -= 1

    return scoreMap
