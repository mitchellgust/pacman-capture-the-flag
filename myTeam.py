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
               first='ValueIterationAgent', second='DefensiveReflexAgent', numTraining=0):
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
        self.cost: int = cost

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


class MDP:
    def __init__(self, width, height, initialValue=' '):
        # TODO: double check this init function - Checked by Mitchell - redundant code removed
        self.width = width
        self.height = height

        self.valueMap = [[initialValue for y in range(height)] for x in range(width)]

        self.livingReward = 0.0
        self.noise = 0.2

    def getStates(self):
        states = []
        for x in range(self.width):
            for y in range(self.height):
                if self.valueMap[x][y] != '#':
                    state = (x, y)
                    states.append(state)
        return states

    def getReward(self, state, action, nextState, gameState, agent):
        x, y = state
        cellValue = self.valueMap[x][y]
        if type(cellValue) == int or type(cellValue) == float:
            return cellValue
        
        
        lastPositionReward = -1
        scaredGhostReward = 20
        foodReward = 30
        ghostReward = -1000
        capsuleReward = 40
        defaultReward = -0.1

        walls = gameState.getWalls()
        food = agent.getFood(gameState).asList()
        capsules = agent.getCapsules(gameState)
        enemyIndexes = agent.getOpponents(gameState)
        observableEnemyIndexes = [
            enemyIndex for enemyIndex in enemyIndexes if gameState.getAgentPosition(enemyIndex)]
        scaredEnemyPositions = [gameState.getAgentPosition(
            enemyIndex) for enemyIndex in observableEnemyIndexes if gameState.getAgentState(enemyIndex).scaredTimer > 0]
        unscaredEnemyPositions = [gameState.getAgentPosition(
            enemyIndex) for enemyIndex in observableEnemyIndexes if
            gameState.getAgentState(enemyIndex).scaredTimer == 0]

        previousObservation = None
        try:
            previousObservation = agent.getPreviousObservation()
        except IndexError:
            pass

        if state in unscaredEnemyPositions:
            return ghostReward
        elif state in [
            agent.getPreviousObservation().getAgentPosition(agent.index) if previousObservation is not None else None]:
            return lastPositionReward
        elif state in food:
            if len(unscaredEnemyPositions) > 0:
                foodRisk = agent.distanceMapToOpenPositions[state]
                return -200 if foodRisk > 1 else foodReward
            else:
                return foodReward
        elif state in walls:
            return None
        elif state in scaredEnemyPositions:
            return scaredGhostReward
        elif state in capsules:
            return capsuleReward
        else:
            return defaultReward

    def getTransitionStatesAndProbs(self, state, action):

        x, y = state

        successors = []

        northState = (self.__isAllowed(y + 1, x) and (x, y + 1)) or state
        westState = (self.__isAllowed(y, x - 1) and (x - 1, y)) or state
        southState = (self.__isAllowed(y - 1, x) and (x, y - 1)) or state
        eastState = (self.__isAllowed(y, x + 1) and (x + 1, y)) or state

        if action == Directions.NORTH or action == Directions.SOUTH:
            if action == Directions.NORTH:
                successors.append((northState, 1 - self.noise))
            else:
                successors.append((southState, 1 - self.noise))

            massLeft = self.noise
            successors.append((westState, massLeft / 2.0))
            successors.append((eastState, massLeft / 2.0))

        if action == Directions.WEST or action == Directions.WEST:
            if action == Directions.WEST:
                successors.append((westState, 1 - self.noise))
            else:
                successors.append((eastState, 1 - self.noise))

            massLeft = self.noise
            successors.append((northState, massLeft / 2.0))
            successors.append((southState, massLeft / 2.0))

        successors = self.__aggregate(successors)

        return successors

    def __aggregate(self, statesAndProbs):
        counter = util.Counter()
        for state, prob in statesAndProbs:
            counter[state] += prob
        newStatesAndProbs = []
        for state, prob in list(counter.items()):
            newStatesAndProbs.append((state, prob))
        return newStatesAndProbs

    def __isAllowed(self, y, x):
        if y < 0 or y >= self.height: return False
        if x < 0 or x >= self.width: return False
        return self.valueMap[x][y] != '#'


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
        for action in ALL_ACTIONS:  # Includes Stop Action
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

        return None  # Failure

    def isRowOrColumnEdge(self, row, column):
        return row == 0 or row == self.mapWidth - 1 or column == 0 or column == self.mapHeight - 1

    def getOpenPositions(self, gameState: GameState):
        openPositions = []
        for i in range(self.mapWidth):
            for j in range(self.mapHeight):
                # Don't Check Position that is a Wall
                if gameState.hasWall(i, j):
                    continue

                # Don't Check Row or Column that is on Edge of Map
                if self.isRowOrColumnEdge(i, j):
                    continue

                # For All Other Positions, Count Surrounding Walls
                wallCount = 0
                positionsToCheck = [(i, j - 1), (i + 1, j), (i, j + 1), (i - 1, j)]  # North, East, South, West

                for x, y in positionsToCheck:
                    if gameState.hasWall(x, y):
                        wallCount += 1

                # If Position is Surrounded by 1 or 0 Walls, Add to List
                if wallCount < 2:
                    openPositions.append((i, j))

        return openPositions

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
            gameState.getAgentPosition(enemyIndex) for enemyIndex in enemyIndexes if
            gameState.getAgentPosition(enemyIndex)]
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
                secondSuccessorsOfEnemy = [successor for first_successor in firstSuccessorsOfEnemy for successor in
                                           self.getSuccessors(first_successor.state, self.wallPositions)]
                thirdSuccessorsOfEnemy = [successor for second_successor in secondSuccessorsOfEnemy for successor in
                                          self.getSuccessors(
                                              second_successor.state, self.wallPositions)]
                successors3AwayFromEnemy = [successor.state for successor in thirdSuccessorsOfEnemy if
                                            self.getMazeDistance(successor.state, closestEnemyPosition) == 3]
                # self.debugDraw(successors3AwayFromEnemy, [1, 0, 0], clear=False)
                successorClosestToCurrentPosition = min(successors3AwayFromEnemy,
                                                        key=lambda x: self.getMazeDistance(currentPosition, x),
                                                        default=self.middleOfMap)
                goalPosition = successorClosestToCurrentPosition

        best_action = self.aStarSearch(
            currentPosition, goalPosition, self.wallPositions, util.manhattanDistance)
        self.currentTarget = goalPosition

        return best_action


class ValueIterationAgent(BaselineAgent):
    def registerInitialState(self, gameState: GameState):
        super().registerInitialState(gameState)
        CaptureAgent.registerInitialState(self, gameState)

        self.mdp = MDP(self.mapWidth, self.mapHeight)
        self.currentPosition = gameState.getAgentPosition(self.index)

        # AKA discount
        self.gamma = 0.9
        self.iterations = 100
        self.values = util.Counter()
        openPositions = self.getOpenPositions(gameState)
        self.distanceMapToOpenPositions = self.getDistanceMapToOpenPositions(
            gameState, openPositions)
        self.runValueIteration(gameState)

    def chooseAction(self, gameState: GameState):
        openPositions = self.getOpenPositions(gameState)
        self.distanceMapToOpenPositions = self.getDistanceMapToOpenPositions(gameState, openPositions)
      
        bestQVal = float("-inf")
        bestAction = None
        self.currentPosition = gameState.getAgentPosition(self.index)

        legalActions = gameState.getLegalActions(self.index)
        for action in legalActions:
            qVal = self.computeQValueFromValues(self.currentPosition, action, gameState)
            print(action, qVal)
            if qVal > bestQVal:
                bestQVal = qVal
                bestAction = action

        print("------")
        return bestAction

    def runValueIteration(self, gameState: GameState):
        allStates = self.mdp.getStates()

        iterationCount = 0
        while iterationCount < self.iterations:
            print("Iteration: ", iterationCount)
            iterationCount += 1
            tempValues = util.Counter()
            for state in allStates:
                legalActions = gameState.getLegalActions(self.index)
                maxQVal = float("-inf")
                for action in legalActions:
                    qValue = self.computeQValueFromValues(state, action, gameState)
                    if qValue > maxQVal:
                        maxQVal = qValue
                tempValues[state] = maxQVal
            self.values = tempValues

    def computeQValueFromValues(self, state, action, gameState):
        qVal = 0

        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)

        for nextState, prob in transitionStatesAndProbs:
            reward = self.mdp.getReward(
                state, action, nextState, gameState, self) + self.gamma * self.values[nextState]
            qVal += prob * reward

        return qVal
