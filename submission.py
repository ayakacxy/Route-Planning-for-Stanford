from typing import List, Tuple

from mapUtil import (
    CityMap,
    computeDistance,
    createStanfordMap,
    locationFromTag,
    makeTag,
)
from util import Heuristic, SearchProblem, State, UniformCostSearch

# BEGIN_YOUR_CODE (You may add some codes here to assist your coding below if you want, but don't worry if you deviate from this.)

# END_YOUR_CODE

# *IMPORTANT* :: A key part of this assignment is figuring out how to model states
# effectively. We've defined a class `State` to help you think through this, with a
# field called `memory`.
#
# As you implement the different types of search problems below, think about what
# `memory` should contain to enable efficient search!
#   > Check out the docstring for `State` in `util.py` for more details and code.

########################################################################################
# Problem 1a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `startLocation` to any location with the specified `endTag`.
    """

    def __init__(self, startLocation: str, endTag: str, cityMap: CityMap):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")这是原代码
        #返回起始状态，即起始位置的状态。
        return State(self.startLocation, None)  # 返回起始位置的状态
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")这是原代码
        #判断当前状态是否为结束状态，即当前位置是否包含结束标签。
        return self.endTag in self.cityMap.tags[state.location]  # 如果当前位置包含结束标签，则返回True，否则返回False
        # END_YOUR_CODE

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        #返回当前状态的后继状态以及到达每个后继状态的代价。
        successors = []
        for nextLocation, distance in self.cityMap.distances[state.location].items():
            nextState = State(nextLocation, None)
            successors.append([nextLocation, nextState, distance])
        return successors
        # 返回一个列表，其中每个元素是一个包含后继位置、对应的状态和到达该状态的代价的元组列表
        # END_YOUR_CODE


########################################################################################
# Problem 1b: Custom -- Plan a Route through Stanford


def getStanfordShortestPathProblem() -> ShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`endTag`. If you prefer, you may create a new map using via
    `createCustomMap()`.

    Run `python mapUtil.py > readableStanfordMap.txt` to dump a file with a list of
    locations and associated tags; you might find it useful to search for the following
    tag keys (amongst others):
        - `landmark=` - Hand-defined landmarks (from `data/stanford-landmarks.json`)
        - `amenity=`  - Various amenity types (e.g., "park", "food")
        - `parking=`  - Assorted parking options (e.g., "underground")
    """
    cityMap = createStanfordMap()

    # Or, if you would rather use a custom map, you can uncomment the following!
    # cityMap = createCustomMap("data/custom.pbf", "data/custom-landmarks".json")

    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    startLocation = str(354802071)#这个参数太难调了，而且有的地图里面的数据只保留到了5位小数,这个实在是尽力了
    endTag = "landmark=AOERC"#这个标签也要硬凑
    #这是短的路线
    # startLocation=str(5676637997)
    # endTag="amenity=parking_entrance"
    return ShortestPathProblem(startLocation, endTag, cityMap)


########################################################################################
# Problem 2a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`.

    Think carefully about what `memory` representation your States should have!
    """
    def __init__(
        self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap
    ):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

        # We want waypointTags to be consistent/canonical (sorted) and hashable (tuple)
        self.waypointTags = tuple(sorted(waypointTags))

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return State(self.startLocation, tuple(1 if tag in self.cityMap.tags[self.startLocation] else 0 for tag in self.waypointTags))
        #使用列表推导式生成一个包含0和1的元组，表示起始位置与路径点之间的关系。如果路径点在起始位置的标签中，则对应位置为1，否则为0。
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        # 检查是否到达终点位置
        endReached = self.endTag in self.cityMap.tags[state.location]
        # 如果尚未到达终点位置，则返回 False
        if not endReached: 
            return False
        # 检查状态记忆中的每个元素，如果有任何一个元素为 0，则返回 False
        for element in state.memory:
            if element == 0: 
                return False
        # 如果已经到达终点位置且状态记忆中的所有元素都为 1，则返回 True
        return True
        # END_YOUR_CODE
    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
        # 初始化空列表用于存储结果
        returnedArr = []
        # 遍历当前位置与相邻位置的距离
        for nextLocation, distance in self.cityMap.distances[state.location].items():
            # 复制当前状态的记忆
            nextState_memory = list(state.memory)            
            # 更新下一个状态的记忆，如果路径点存在于下一个位置的标签中，则将对应位置设置为1
            for i in range(len(self.waypointTags)):
                if self.waypointTags[i] in self.cityMap.tags[nextLocation]:
                    nextState_memory[i] = 1            
            # 将记忆转换为元组
            nextState_memory = tuple(nextState_memory)           
            # 创建下一个状态并将其与相邻位置和距离一起添加到结果列表中
            returnedArr.append((nextLocation, State(nextLocation, nextState_memory), distance))
        # 返回结果列表
        return returnedArr
        # END_YOUR_CODE


########################################################################################
# Problem 2b: Custom -- Plan a Route with Unordered Waypoints through Stanford


def getStanfordWaypointsShortestPathProblem() -> WaypointsShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`waypointTags`/`endTag`.

    Similar to Problem 1b, use `readableStanfordMap.txt` to identify potential
    locations and tags.
    """
    cityMap = createStanfordMap()
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)   
    #raise Exception("Not implemented yet")
    # startLocation = str(354802071)#复制一下
    # endTag = "landmark=AOERC"
    waypointTags = ["kerb=lowered", "landmark=bookstore", "landmark=evgr_a", "entrance=yes", "barrier=gate"]#随便写一点
    #这是短的路线
    startLocation=str(5676637997)
    endTag="amenity=parking_entrance"
    #END_YOUR_CODE
    return WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)


########################################################################################
# Problem 3a: A* to UCS reduction

# Turn an existing SearchProblem (`problem`) you are trying to solve with a
# Heuristic (`heuristic`) into a new SearchProblem (`newSearchProblem`), such
# that running uniform cost search on `newSearchProblem` is equivalent to
# running A* on `problem` subject to `heuristic`.
#
# This process of translating a model of a problem + extra constraints into a
# new instance of the same problem is called a reduction; it's a powerful tool
# for writing down "new" models in a language we're already familiar with.


def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def startState(self) -> State:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            #raise Exception("Not implemented yet")
            return problem.startState()
            # END_YOUR_CODE

        def isEnd(self, state: State) -> bool:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            #raise Exception("Not implemented yet")
            return problem.isEnd(state)
            # END_YOUR_CODE

        def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
            #raise Exception("Not implemented yet")
            # 获取后继状态及其相关成本的列表
            succList = problem.successorsAndCosts(state)
            # 对于每个元素进行处理
            for index, element in enumerate(succList):
                # 将元素转换为列表以便进行修改
                element = list(element)
                # 计算启发式评估值的差异并添加到成本中
                element[2] += heuristic.evaluate(element[1]) - heuristic.evaluate(state)
                # 将修改后的元素转换回元组形式
                succList[index] = tuple(element)
            # 返回修改后的后继状态列表
            return succList
            # END_YOUR_CODE

    return NewSearchProblem()


########################################################################################
# Problem 3b: "straight-line" heuristic for A*


class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
        > Hint: you might consider using `computeDistance` defined in `mapUtil.py`
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

        # Precompute
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        # 使用 for 循环遍历地图的所有位置
        self.endLocations = []
        # 遍历地图中的每个位置
        for location in self.cityMap.geoLocations.keys():
            # 检查当前位置的标签是否包含特定标签
            if endTag in self.cityMap.tags[location]:
                # 如果包含特定标签，则将该位置添加到self.endLocations列表中
                self.endLocations.append(location)
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        # 初始化一个空列表，用于存储当前状态到所有终点位置的距离
        distances = []
        # 遍历所有终点位置
        for endLocation in self.endLocations:
            # 计算当前状态到当前终点位置的距离
            distance = computeDistance(self.cityMap.geoLocations[state.location], self.cityMap.geoLocations[endLocation])
            # 将距离添加到列表中
            distances.append(distance)
        # 返回距离列表中的最大值
        return max(distances)
        # END_YOUR_CODE


########################################################################################
# Problem 3c: "no waypoints" heuristic for A*


class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `startLocation` to any location with `endTag`,
    ignoring all waypoints.
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        # Precompute
        # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
        #直接重复问题1的解决方案
        #raise Exception("Not implemented yet")
        # 定义一个反向最短路径问题类，用于预计算到达具有指定终点标签的每个位置的最短路径成本
        class ReverseShortestPathProblem(SearchProblem):
            def startState(self) -> State:
                # 返回特殊的结束状态
                return State("Special END state", None)

            def isEnd(self,state: State) -> bool:
                # 对于每个状态都返回 False，因为没有有效的结束状态
                # 但是看着这个灰色的很不爽
                _ = state  # 对state参数进行一个简单的操作，比如赋值给一个临时变量，这样就不灰了
                return False

            def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
                # 如果当前位置是特殊的结束状态，则返回所有具有期望终点标签的位置和成本为 0
                # 否则，返回当前位置的所有后继位置及其对应的距离，
                returnedArr = []
                if state.location == "Special END state":
                    for endLocation in cityMap.geoLocations.keys():
                        if endTag in cityMap.tags[endLocation]:
                            returnedArr.append((endLocation, State(endLocation, None), 0))
                else:
                    for nextLocation, distance in cityMap.distances[state.location].items():
                        returnedArr.append((nextLocation, State(nextLocation, None), distance))
                return returnedArr

        # 创建反向最短路径问题的实例并使用 UCS 求解
        reverse_shortest_path_problem = ReverseShortestPathProblem()
        uniform_cost_search = UniformCostSearch(verbose=0)
        uniform_cost_search.solve(reverse_shortest_path_problem)

        # 将 UCS 求解得到的过去成本存储为启发式函数
        self.heuristicFunction = uniform_cost_search.pastCosts
    # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        return self.heuristicFunction[state.location]
        # END_YOUR_CODE
