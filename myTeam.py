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

  def registerInitialState(self, gameState):
    # For common variables
    
    # Initialise Ghost Indexes
    self.ghosts = self.getOpponents(gameState)
    # Get Ghost Positions
    self.ghostPositions = []
    for ghost in self.ghosts:
      self.ghostPositions.append(gameState.getAgentPosition(ghost))

    # # Initialise the start position
    # self.startPosition = gameState.getAgentPosition(self.index)

    # # Initialise Food Positions
    # self.foodPositions = self.getFood(gameState).asList()

    # # Initialise Wall Positions
    # self.wallPositions = gameState.getWalls().asList()

    # # Initialise Capsule Positions and Capsules
    # self.capsulePositions = self.getCapsules(gameState)

    # # Get Legal Actions and Closest
    # self.legalActions = gameState.getLegalActions(self.index)

    # # Get Score
    # self.score = self.getScore(gameState)
  
  def aStarSearch(self, start_position, goal_position, walls, heuristic):
    
    def getSuccessors(state, walls):
      successors = []
      for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
        x, y = state
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        if (nextx, nexty) not in walls:
          new_node = Node((nextx, nexty), action, 1)
          successors.append(new_node)
      return successors
      
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
      for successor in getSuccessors(state, walls):
          successorPath = path + [successor.path]
          successorCost = cost + successor.cost
          successorNode = Node(successor.state, successorPath, successorCost)
          
          frontier.push(successorNode)

    return None # Failure

class OffensiveReflexAgent(BaselineAgent):

  def registerInitialState(self, gameState):
    super().registerInitialState(gameState)

    CaptureAgent.registerInitialState(self, gameState)
  
  def chooseAction(self, gameState):
    # Is the agent scared?
    for ghost in self.ghosts:
      if gameState.getAgentState(ghost).scaredTimer > 0:
        print("Scared Ghost Found")

    return 'Stop'


class DefensiveReflexAgent(BaselineAgent):
  def registerInitialState(self, gameState):
    super().registerInitialState(gameState)
    
    self.start = gameState.getAgentPosition(self.index)
    self.walls = gameState.getWalls().asList()
    self.current_target = None
    self.scaredGhosts = []
    
    # TODO: Initialize variables we need here:
    # positions to food, enemy pacman, capsules and the entrances of 
    # the ally's side of the maze
    
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    # TODO: Get information about the gameState
    # Is the agent scared? Should we stay away from the enemy but as close as possible?
    # Are there any enemies within 5 steps of the agent? Chase them!
    # Is there any food that was present in the last observation that 
    # was eaten? Go to that location.
    # 
    # Once we have decided what to do, we can call aStarSearch to find the best action
    
    # Information about the gameState and current agent
    current_position = gameState.getAgentPosition(self.index)
    last_state = self.getPreviousObservation()
    goal_position = self.current_target if self.current_target else self.start
    
    # Move to the closest food that was eaten my an enemy pacman from the last iteration
    if last_state: 
      last_state_foods = self.getFoodYouAreDefending(last_state).asList()
      current_state_foods = self.getFoodYouAreDefending(gameState).asList()
      foods_eaten_since_last_state = [food for food in last_state_foods if food not in current_state_foods]
      
      if foods_eaten_since_last_state:
        goal_position = min(foods_eaten_since_last_state,
                            key=lambda x: self.getMazeDistance(current_position, x))
        
    # Function 2
    # When Idle - Put Agent near the middle of the maze, priorititising the location to be in a conjestion of food

    # Function 3
    # When Scared - Move away from the enemy pacman but stay as close as possible
    # Reset Scared
    self.scaredGhosts = []
    # Get Scared Ghosts
    for ghost in self.ghosts:
      if gameState.getAgentState(ghost).scaredTimer > 0:
        self.scaredGhosts.append(ghost)
        print("Scared Ghost Found")

 
    best_action = self.aStarSearch(
        current_position, goal_position, self.walls, util.manhattanDistance)
    self.current_target = goal_position   
    
    return best_action
