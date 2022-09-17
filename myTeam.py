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


from lib2to3.pytree import Node
from turtle import position
from captureAgents import CaptureAgent
import random, time, util
from distanceCalculator import Distancer
from game import Directions
import game

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

class BaselineAgent(CaptureAgent):

  nullHeuristic = 0

  class Node:
    def __init__(self, state, path, cost):
      self.state = state
      self.path = path
      self.cost = cost
    
    @property
    def getState(self):
      return self.state
    
    @property
    def getPath(self):
      return self.path
    
    @property
    def getCost(self):
      return self.cost

  def aStarSearch(problem, heuristic=nullHeuristic):
    def priorityQueueFunction(node: Node): 
      # f(n) = g(n) + h(n) (priority = cost + estimatedCost)
      return node.cost + heuristic(node.state, problem)

    priorityQueue = util.PriorityQueueWithFunction(priorityQueueFunction)

    initialNode = Node(problem.getStartState(), [], 0)
    frontier = priorityQueue
    frontier.push(initialNode)

    reached = []
    while not frontier.isEmpty():
      state, path, cost = frontier.pop()
      
      if problem.isGoalState(state):
          return path

      if state in reached:
          continue
      
      reached.append(state)

      successor: Node
      for successor in problem.getSuccessors(state):
          successorPath = path + [successor.path]
          successorCost = cost + successor.cost
          successorNode = (successor.state, successorPath, successorCost)
          
          frontier.push(successorNode)

    return None # Failure


class OffensiveReflexAgent(BaselineAgent):
  def registerInitialState(self, gameState):
    # Initialise the start position
    self.startPosition = gameState.getAgentPosition(self.index)
    
    # Initialise Ghost Indexes
    self.ghosts = self.getOpponents(gameState)
    # Get Ghost Positions
    self.ghostPositions = []
    for ghost in self.ghosts:
      self.ghostPositions.append(gameState.getAgentPosition(ghost))

    # Initialise Food Positions
    self.foodPositions = self.getFood(gameState).asList()

    # Initialise Wall Positions
    self.wallPositions = gameState.getWalls().asList()

    # Initialise Capsule Positions and Capsules
    self.capsulePositions = self.getCapsules(gameState)

    # Get Legal Actions and Closest
    self.legalActions = gameState.getLegalActions(self.index)

    # Get Score
    self.score = self.getScore(gameState)

    # Print variables
    print("Start: ", self.startPosition)
    print("Ghosts: ", self.ghostPositions)
    print("Food: ", self.foodPositions)
    print("Walls: ", self.wallPositions)
    print("Capsules: ", self.capsulePositions)
    print("Legal Actions: ", self.legalActions)
    print("Score: ", self.score)

    CaptureAgent.registerInitialState(self, gameState)
  
  def chooseAction(self, gameState):
    # Is the agent scared?
    for ghost in self.ghosts:
      if gameState.getAgentState(ghost).scaredTimer > 0:
        print("Scared Ghost Found")

    return 0

  def rewardFunction(self, position):
    reward = -1
    if position in self.foodPositions:
      reward = 5
    if position in self.capsulePositions:
      reward = 10
    if position in self.ghostPositions:
      reward = -20
    return reward

class DefensiveReflexAgent(BaselineAgent):
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    
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
    return
