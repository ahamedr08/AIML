from collections import deque

def tower_of_hanoi_dfs(n, source, auxiliary, destination):
    if n == 1:
        print(f"Move disc 1 from {source} to {destination}")
        return

    tower_of_hanoi_dfs(n - 1, source, destination, auxiliary)
    print(f"Move disc {n} from {source} to {destination}")
    tower_of_hanoi_dfs(n - 1, auxiliary, source, destination)

def tower_of_hanoi_bfs(n, source, destination):
    class State:
        def __init__(self, pegs, steps=""):
            self.pegs = pegs
            self.steps = steps

    initial_state = State([[i for i in range(n, 0, -1)], [], []])
    visited = set()
    queue = deque([initial_state])

    while queue:
        current_state = queue.popleft()

        if current_state.pegs[2] == [i for i in range(n, 0, -1)]:
            return current_state.steps

        for source in range(3):
            for destination in range(3):
                if source != destination and current_state.pegs[source]:
                    new_pegs = [peg[:] for peg in current_state.pegs]
                    disc = new_pegs[source].pop()
                    new_pegs[destination].append(disc)

                    new_state = State(new_pegs, current_state.steps + f"Move disc {disc} from {chr(65+source)} to {chr(65+destination)}\n")
                    
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append(new_state)

    return "No solution found."

def main():
    n = int(input("Enter the number of discs: "))
    
    print("\nBFS Approach:")
    bfs_result = tower_of_hanoi_bfs(n, 'A', 'C')
    print(bfs_result)

    print("\nDFS Approach:")
    tower_of_hanoi_dfs(n, 'A', 'B', 'C')

if __name__ == "__main__":
    main()
