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
      
class BaselineAgent(CaptureAgent):

  def registerInitialState(self, gameState: GameState):
    # For common variables
    mapWidth = gameState.data.layout.width
    mapHeight = gameState.data.layout.height
    midWidth = int(mapWidth / 2)
    midHeight = int(mapHeight / 2)
    
    isRed = self.index in gameState.getRedTeamIndices()
    self.isRed = isRed
    offset = -1 if isRed else 0
  
    entrancePositions = []
    middleOfMap = (midWidth + offset, midHeight)
    if gameState.hasWall(middleOfMap[0], middleOfMap[1]):
      candidates = []
      for i in range(mapHeight):
        if not gameState.hasWall(midWidth + offset, i) and not gameState.hasWall(midWidth, i):
          entrancePositions.append((midWidth + offset, i))

        if not gameState.hasWall(middleOfMap[0], i):
          candidates.append((middleOfMap[0], i))
      middleOfMap = min(candidates, key=lambda x: abs(x[1] - midHeight))

    self.entrancePositions = entrancePositions
    self.middleOfMap = middleOfMap
    
  def getSuccessors(self, state, walls):
      successors = []
      for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
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

      # State((x,y), numFood))
      if state == goal_position:
        if not path:
          return 'Stop'
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

  def registerInitialState(self, gameState: GameState):
    super().registerInitialState(gameState)
    CaptureAgent.registerInitialState(self, gameState)
  
  def chooseAction(self, gameState: GameState):
    # Is the agent scared?
    for enemy in self.getOpponents(gameState):
      if gameState.getAgentState(enemy).scaredTimer > 0:
        # print("Scared Ghost Found")
        pass

    return 'Stop'


class DefensiveReflexAgent(BaselineAgent):
  def registerInitialState(self, gameState: GameState):
    super().registerInitialState(gameState)
    
    self.start = gameState.getAgentPosition(self.index)
    self.walls = gameState.getWalls().asList()
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
    self.debugDraw(self.entrancePositions, [0, 1, 0], clear=True)
    
    # Information about the gameState and current agent
    currentPosition = gameState.getAgentPosition(self.index)
    lastState = self.getPreviousObservation()
    goalPosition = self.currentTarget if self.currentTarget else self.middleOfMap 
    enemyIndexes = self.getOpponents(gameState)
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
        firstSuccessorsOfEnemy = self.getSuccessors(closestEnemyPosition, self.walls)
        secondSuccessorsOfEnemy = [successor for first_successor in firstSuccessorsOfEnemy for successor in self.getSuccessors(first_successor.state, self.walls)]
        thirdSuccessorsOfEnemy = [successor for second_successor in secondSuccessorsOfEnemy for successor in self.getSuccessors(second_successor.state, self.walls)]
        successors3AwayFromEnemy = [successor.state for successor in thirdSuccessorsOfEnemy if self.getMazeDistance(successor.state, closestEnemyPosition) == 3]
        # self.debugDraw(successors3AwayFromEnemy, [1, 0, 0], clear=False)
        successorClosestToCurrentPosition = min(successors3AwayFromEnemy, key=lambda x: self.getMazeDistance(currentPosition, x), default=self.middleOfMap)
        goalPosition = successorClosestToCurrentPosition
        
  
    best_action = self.aStarSearch(
        currentPosition, goalPosition, self.walls, util.manhattanDistance)
    self.currentTarget = goalPosition   
    
    return best_action
