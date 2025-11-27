import random
import typing
from collections import deque
import heapq


def info() -> typing.Dict:
    print("INFO")
    return {
        "apiversion": "1",
        "author": "Aggressive Snake",
        "color": "#FF0000",
        "head": "default",
        "tail": "default",
    }


def start(game_state: typing.Dict):
    print("GAME START")


def end(game_state: typing.Dict):
    print("GAME OVER\n")


def get_next_position(head: typing.Dict, direction: str) -> typing.Dict:
    """Calculate next position given current head and direction"""
    x, y = head["x"], head["y"]
    if direction == "up":
        return {"x": x, "y": y + 1}
    elif direction == "down":
        return {"x": x, "y": y - 1}
    elif direction == "left":
        return {"x": x - 1, "y": y}
    elif direction == "right":
        return {"x": x + 1, "y": y}
    return head


def is_out_of_bounds(pos: typing.Dict, board_width: int, board_height: int) -> bool:
    """Check if position is outside board boundaries"""
    return pos["x"] < 0 or pos["x"] >= board_width or pos["y"] < 0 or pos["y"] >= board_height


def count_adjacent_walls(pos: typing.Dict, board_width: int, board_height: int) -> int:
    """Count how many walls are adjacent to this position"""
    walls = 0
    if pos["x"] == 0:
        walls += 1
    if pos["x"] == board_width - 1:
        walls += 1
    if pos["y"] == 0:
        walls += 1
    if pos["y"] == board_height - 1:
        walls += 1
    return walls


def count_safe_neighbors(pos: typing.Dict, game_state: typing.Dict, snake_positions: set) -> int:
    """Count how many safe adjacent squares exist from this position"""
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]

    directions = ["up", "down", "left", "right"]
    safe_count = 0

    for direction in directions:
        next_pos = get_next_position(pos, direction)

        if is_out_of_bounds(next_pos, board_width, board_height):
            continue
        if (next_pos["x"], next_pos["y"]) in snake_positions:
            continue

        safe_count += 1

    return safe_count


def manhattan_distance(pos1: typing.Dict, pos2: typing.Dict) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1["x"] - pos2["x"]) + abs(pos1["y"] - pos2["y"])


def astar_distance(start: typing.Dict, goal: typing.Dict, game_state: typing.Dict, snake_positions: set) -> typing.Optional[int]:
    """
    Calculate actual path distance using A* pathfinding.
    Returns path length or None if no path exists.
    """
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]

    # Priority queue: (f_score, g_score, position)
    open_set = []
    heapq.heappush(open_set, (0, 0, (start["x"], start["y"])))

    # Track best g_score for each position
    g_scores = {(start["x"], start["y"]): 0}

    # Track visited nodes
    closed_set = set()

    directions = ["up", "down", "left", "right"]

    while open_set:
        f_score, g_score, current_tuple = heapq.heappop(open_set)
        current = {"x": current_tuple[0], "y": current_tuple[1]}

        # Check if we reached the goal
        if current["x"] == goal["x"] and current["y"] == goal["y"]:
            return g_score

        # Skip if already visited
        if current_tuple in closed_set:
            continue

        closed_set.add(current_tuple)

        # Explore neighbors
        for direction in directions:
            neighbor = get_next_position(current, direction)
            neighbor_tuple = (neighbor["x"], neighbor["y"])

            # Skip if out of bounds
            if is_out_of_bounds(neighbor, board_width, board_height):
                continue

            # Skip if hits a snake body (but allow goal position)
            if neighbor_tuple in snake_positions:
                if neighbor["x"] != goal["x"] or neighbor["y"] != goal["y"]:
                    continue

            # Calculate tentative g_score
            tentative_g = g_score + 1

            # Skip if we've found a better path to this neighbor
            if neighbor_tuple in g_scores and tentative_g >= g_scores[neighbor_tuple]:
                continue

            # This is the best path to neighbor so far
            g_scores[neighbor_tuple] = tentative_g

            # Calculate f_score (g + heuristic)
            h_score = manhattan_distance(neighbor, goal)
            f_score = tentative_g + h_score

            heapq.heappush(open_set, (f_score, tentative_g, neighbor_tuple))

    # No path found
    return None


def flood_fill(start: typing.Dict, game_state: typing.Dict, avoid_positions: set) -> int:
    """Count reachable spaces from start position using BFS"""
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]

    visited = set()
    queue = deque([start])
    visited.add((start["x"], start["y"]))

    directions = ["up", "down", "left", "right"]

    while queue:
        current = queue.popleft()

        for direction in directions:
            next_pos = get_next_position(current, direction)
            pos_tuple = (next_pos["x"], next_pos["y"])

            if pos_tuple in visited:
                continue
            if is_out_of_bounds(next_pos, board_width, board_height):
                continue
            if pos_tuple in avoid_positions:
                continue

            visited.add(pos_tuple)
            queue.append(next_pos)

    return len(visited)


def calculate_voronoi(game_state: typing.Dict, snake_positions: set) -> typing.Dict:
    """
    Calculate Voronoi territories - which cells each snake can reach first.
    Returns dict with snake IDs as keys and territory counts as values.
    """
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]

    # Track which snake reaches each cell first and at what distance
    cell_owner = {}  # (x, y) -> snake_id
    cell_distance = {}  # (x, y) -> distance

    # Initialize BFS from all snake heads simultaneously
    queue = deque()
    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id")
        if not snake_id or len(snake.get("body", [])) == 0:
            continue

        head = snake["body"][0]
        head_tuple = (head["x"], head["y"])
        queue.append((head_tuple, snake_id, 0))
        cell_owner[head_tuple] = snake_id
        cell_distance[head_tuple] = 0

    directions = ["up", "down", "left", "right"]

    # BFS to fill the board
    while queue:
        current_tuple, snake_id, distance = queue.popleft()
        current = {"x": current_tuple[0], "y": current_tuple[1]}

        for direction in directions:
            next_pos = get_next_position(current, direction)
            next_tuple = (next_pos["x"], next_pos["y"])

            # Skip out of bounds
            if is_out_of_bounds(next_pos, board_width, board_height):
                continue

            # Skip snake bodies
            if next_tuple in snake_positions:
                continue

            new_distance = distance + 1

            # If unclaimed or we can reach it faster/same time
            if next_tuple not in cell_distance or new_distance <= cell_distance[next_tuple]:
                # If same distance, it's contested (could mark as neutral, but we'll give it to first)
                if next_tuple not in cell_distance:
                    cell_owner[next_tuple] = snake_id
                    cell_distance[next_tuple] = new_distance
                    queue.append((next_tuple, snake_id, new_distance))

    # Count territories
    territory_count = {}
    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id")
        if snake_id:
            territory_count[snake_id] = 0

    for pos, owner_id in cell_owner.items():
        territory_count[owner_id] = territory_count.get(owner_id, 0) + 1

    return territory_count


def get_game_phase(game_state: typing.Dict) -> str:
    """Determine current game phase: early, late, or 1v1"""
    alive_snakes = len(game_state["board"]["snakes"])

    if alive_snakes == 2:
        return "1v1"
    elif alive_snakes <= 3:
        return "late"
    else:
        return "early"


def get_opponent_heads(game_state: typing.Dict) -> typing.List:
    """Get positions and lengths of opponent snake heads"""
    my_id = game_state["you"].get("id", "")
    opponent_heads = []

    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id", "")
        if snake_id != my_id and len(snake.get("body", [])) > 0:
            opponent_heads.append({
                "pos": snake["body"][0],
                "length": len(snake["body"]),
                "id": snake_id
            })

    return opponent_heads


def get_opponent_stats(game_state: typing.Dict) -> typing.Dict:
    """Get stats about opponents: max length, avg length, count"""
    my_id = game_state["you"].get("id", "")
    opponent_lengths = []

    for snake in game_state["board"]["snakes"]:
        snake_id = snake.get("id", "")
        if snake_id != my_id and len(snake.get("body", [])) > 0:
            opponent_lengths.append(len(snake["body"]))

    if len(opponent_lengths) == 0:
        return {"max_length": 0, "avg_length": 0, "count": 0}

    return {
        "max_length": max(opponent_lengths),
        "avg_length": sum(opponent_lengths) / len(opponent_lengths),
        "count": len(opponent_lengths)
    }


def is_head_to_head_safe(pos: typing.Dict, my_length: int, opponent_heads: typing.List) -> bool:
    """Check if position is safe from head-to-head collisions"""
    for opponent in opponent_heads:
        dist = manhattan_distance(pos, opponent["pos"])
        # Only avoid if we're adjacent (distance 1) and opponent is same size or larger
        # This is where actual head-to-head collision can happen next turn
        if dist == 1 and opponent["length"] >= my_length:
            return False
    return True


def get_all_snake_positions(game_state: typing.Dict) -> set:
    """Get all positions occupied by snake bodies that will still be there next turn"""
    positions = set()

    for snake in game_state["board"]["snakes"]:
        body = snake.get("body", [])

        if len(body) == 0:
            continue

        for i, segment in enumerate(body):
            # Skip head (it will move to a new position) - index 0
            if i == 0:
                continue

            # Include all body segments including tail
            # Conservative approach: assume tail might not move (snake could eat)
            # Better to avoid a position that might be safe than collide
            pos_tuple = (segment["x"], segment["y"])
            positions.add(pos_tuple)

    return positions


def predict_opponent_moves(opponent_head: typing.Dict, opponent_length: int, game_state: typing.Dict, snake_positions: set) -> typing.Dict:
    """Predict where opponent might move and score their options"""
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]

    opponent_moves = {}
    directions = ["up", "down", "left", "right"]

    for direction in directions:
        next_pos = get_next_position(opponent_head, direction)
        next_pos_tuple = (next_pos["x"], next_pos["y"])

        # Same safety checks opponent would use
        if is_out_of_bounds(next_pos, board_width, board_height):
            continue
        if next_pos_tuple in snake_positions:
            continue

        # Score based on space they'd have
        spaces = flood_fill(next_pos, game_state, snake_positions)
        opponent_moves[direction] = {
            "pos": next_pos,
            "spaces": spaces
        }

    return opponent_moves


def has_kill_move(my_head: typing.Dict, my_length: int, opponent: typing.Dict, game_state: typing.Dict, snake_positions: set) -> typing.Optional[str]:
    """Check if we can force opponent into zero safe moves - winning move detection"""
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    directions = ["up", "down", "left", "right"]

    for my_dir in directions:
        my_next = get_next_position(my_head, my_dir)
        my_next_tuple = (my_next["x"], my_next["y"])

        # Skip if our move is invalid
        if is_out_of_bounds(my_next, board_width, board_height):
            continue
        if my_next_tuple in snake_positions:
            continue

        # Simulate our move - add our new position
        new_positions = snake_positions.copy()
        new_positions.add(my_next_tuple)

        # Check if opponent has ANY safe moves left
        opponent_safe_moves = 0
        for opp_dir in directions:
            opp_next = get_next_position(opponent["pos"], opp_dir)
            opp_next_tuple = (opp_next["x"], opp_next["y"])

            # Check if out of bounds
            if is_out_of_bounds(opp_next, board_width, board_height):
                continue

            # Check if hits a snake body (excluding our new head position)
            if opp_next_tuple in snake_positions:
                continue

            # Check for head-to-head collision with our new position
            if opp_next_tuple == my_next_tuple:
                # Only safe for opponent if they're longer or equal (they win/tie)
                if opponent["length"] >= my_length:
                    opponent_safe_moves += 1
                continue

            # Check if they'd have enough space after this move
            opp_spaces = flood_fill(opp_next, game_state, new_positions)
            if opp_spaces >= opponent["length"]:
                opponent_safe_moves += 1

        if opponent_safe_moves == 0:
            return my_dir  # WINNING MOVE!

    return None


def move(game_state: typing.Dict) -> typing.Dict:
    my_head = game_state["you"]["body"][0]
    my_length = len(game_state["you"]["body"])
    my_health = game_state["you"]["health"]
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    food = game_state["board"]["food"]

    print(f"Turn {game_state['turn']}: Head at ({my_head['x']}, {my_head['y']}), Board: {board_width}x{board_height}")

    game_phase = get_game_phase(game_state)
    opponent_heads = get_opponent_heads(game_state)
    opponent_stats = get_opponent_stats(game_state)

    # Get all snake body positions that will still be there next turn
    snake_positions = get_all_snake_positions(game_state)

    length_target = opponent_stats['max_length'] + 2
    need_length = my_length < length_target

    # Calculate Voronoi territories
    my_id = game_state["you"].get("id", "")
    voronoi_territories = calculate_voronoi(game_state, snake_positions)
    my_territory = voronoi_territories.get(my_id, 0)

    # 1v1 KILL MOVE DETECTION - highest priority
    if game_phase == "1v1" and len(opponent_heads) > 0:
        opponent = opponent_heads[0]
        kill_move = has_kill_move(my_head, my_length, opponent, game_state, snake_positions)
        if kill_move:
            print(f"  KILL MOVE DETECTED: {kill_move}")
            return {"move": kill_move}

    directions = ["up", "down", "left", "right"]
    move_scores = {}

    my_neck = game_state["you"]["body"][1] if len(game_state["you"]["body"]) > 1 else None
    opponents_small = opponent_stats['max_length'] <= 6

    for direction in directions:
        next_pos = get_next_position(my_head, direction)
        next_pos_tuple = (next_pos["x"], next_pos["y"])
        score = 0

        # SURVIVAL CHECKS - immediate disqualification

        # Check reversing into neck
        if my_neck and next_pos["x"] == my_neck["x"] and next_pos["y"] == my_neck["y"]:
            move_scores[direction] = -10000
            continue

        # Check walls
        if is_out_of_bounds(next_pos, board_width, board_height):
            move_scores[direction] = -10000
            continue

        # Check snake bodies
        if next_pos_tuple in snake_positions:
            move_scores[direction] = -10000
            continue

        # Check head-to-head collisions
        if not is_head_to_head_safe(next_pos, my_length, opponent_heads):
            move_scores[direction] = -10000
            continue

        # TRAP DETECTION - check if we're walking into a corner or tight spot
        adjacent_walls = count_adjacent_walls(next_pos, board_width, board_height)
        safe_neighbors = count_safe_neighbors(next_pos, game_state, snake_positions)

        # Reduce trap penalties only in early game when opponents are small
        trap_multiplier = 0.3 if (game_phase == "early" and opponents_small) else 1.0

        # Corner = 2 walls, tight spot = limited safe neighbors
        if adjacent_walls >= 2:
            score -= 1500 * trap_multiplier

        if safe_neighbors <= 1:
            score -= 1000 * trap_multiplier
        elif safe_neighbors == 2:
            score -= 200 * trap_multiplier

        # SPACE CONTROL - flood fill to count reachable spaces
        reachable_spaces = flood_fill(next_pos, game_state, snake_positions)

        # Must have enough space to survive
        if reachable_spaces < my_length:
            score -= 5000
        else:
            # Reduce space control weight in early game when we need food
            if game_phase == "early" and need_length:
                score += reachable_spaces * 0.5  # Almost no weight - food is everything
            elif game_phase == "early":
                score += reachable_spaces * 2  # Very low weight in early game
            else:
                score += reachable_spaces * 10  # Normal weight for late game

        # VORONOI TERRITORY CONTROL - only in late game (too slow for early game)
        if game_phase in ["late", "1v1"]:
            # Create a simulated game state with our head at next_pos
            simulated_game_state = {
                "board": {
                    "width": board_width,
                    "height": board_height,
                    "snakes": []
                }
            }

            # Add our snake with new head position
            simulated_my_body = [next_pos] + game_state["you"]["body"][:-1]  # Move head, remove tail
            simulated_game_state["board"]["snakes"].append({
                "id": my_id,
                "body": simulated_my_body
            })

            # Add opponent snakes (unchanged for now - simplified simulation)
            for snake in game_state["board"]["snakes"]:
                snake_id = snake.get("id", "")
                if snake_id and snake_id != my_id:
                    simulated_game_state["board"]["snakes"].append(snake)

            # Calculate voronoi after this move
            simulated_snake_positions = get_all_snake_positions(simulated_game_state)
            future_voronoi = calculate_voronoi(simulated_game_state, simulated_snake_positions)
            future_my_territory = future_voronoi.get(my_id, 0)

            # Territory gain/loss
            territory_change = future_my_territory - my_territory

            # Calculate how much we're cutting off opponents
            opponent_territory_loss = 0
            for opp_id, opp_territory in voronoi_territories.items():
                if opp_id and opp_id != my_id:
                    future_opp_territory = future_voronoi.get(opp_id, 0)
                    opponent_territory_loss += (opp_territory - future_opp_territory)

            # Reward territory gain
            score += territory_change * 15  # Strong weight in late game
            score += opponent_territory_loss * 10  # Reward cutting off opponents

        # 1v1 AGGRESSIVE SPACE CUTTING - trap opponent when ahead
        if game_phase == "1v1" and len(opponent_heads) > 0:
            opponent = opponent_heads[0]

            if my_length > opponent["length"]:
                # Calculate opponent's reachable space before our move
                opponent_reachable_before = flood_fill(opponent["pos"], game_state, snake_positions)

                # Simulate our move
                simulated_positions = snake_positions.copy()
                simulated_positions.add(next_pos_tuple)

                # Calculate opponent's reachable space after our move
                opponent_reachable_after = flood_fill(opponent["pos"], game_state, simulated_positions)

                # Reward moves that cut off opponent space
                space_reduction = opponent_reachable_before - opponent_reachable_after
                score += space_reduction * 50

                # CRITICAL: Massive bonus if we can trap them (reduce space below their length)
                if opponent_reachable_after < opponent["length"]:
                    score += 5000  # Near-kill move
                    print(f"  Space cutting: opponent space={opponent_reachable_after}, opponent_length={opponent['length']}")

        # 1v1 TAIL CHASING - our tail is safe when ahead
        if game_phase == "1v1" and len(opponent_heads) > 0:
            opponent = opponent_heads[0]
            my_tail = game_state["you"]["body"][-1]
            tail_dist = manhattan_distance(next_pos, my_tail)

            if my_length > opponent["length"] + 3:
                # When significantly ahead, follow our tail - it's guaranteed safe
                score -= tail_dist * 8

            # Also chase opponent's tail - they can't trap us if we're longer
            if my_length > opponent["length"]:
                opponent_body = None
                for snake in game_state["board"]["snakes"]:
                    if snake.get("id") == opponent["id"]:
                        opponent_body = snake.get("body", [])
                        break

                if opponent_body and len(opponent_body) > 0:
                    opponent_tail = opponent_body[-1]
                    opp_tail_dist = manhattan_distance(next_pos, opponent_tail)
                    score -= opp_tail_dist * 5  # Chase them

        # FOOD SEEKING - aggressive early growth, strategic late game
        if len(food) > 0:
            # Use Manhattan distance for speed
            food_distances = [manhattan_distance(next_pos, f) for f in food]
            closest_food_dist = min(food_distances) if food_distances else 999

            # Reduce food seeking if this move is into a trap
            food_weight_multiplier = 1.0
            if adjacent_walls >= 2 or safe_neighbors <= 1:
                if game_phase == "early" and opponents_small:
                    food_weight_multiplier = 0.8  # Only reduce by 20% when safe
                else:
                    food_weight_multiplier = 0.5  # Reduce by 50% when risky

            # EARLY GAME: hyper-aggressive growth until we have length advantage
            if game_phase == "early":
                if need_length:
                    # MAXIMUM food seeking when behind in length - grow fast!
                    score -= closest_food_dist * 200 * food_weight_multiplier
                    score += (100 - my_health) * 15
                    # Extra massive bonus for being shorter than opponents
                    if my_length <= opponent_stats['avg_length']:
                        score -= closest_food_dist * 120 * food_weight_multiplier
                    # Even more if significantly behind
                    if my_length < opponent_stats['max_length'] - 2:
                        score -= closest_food_dist * 80 * food_weight_multiplier
                elif my_health < 50:
                    # Have length advantage but need health
                    score -= closest_food_dist * 100 * food_weight_multiplier
                    score += (100 - my_health) * 8
                else:
                    # Have length advantage and good health - still seek food
                    score -= closest_food_dist * 50 * food_weight_multiplier

            # LATE GAME: only seek food when necessary
            elif game_phase in ["late", "1v1"]:
                if my_health < 35:
                    # Desperate for food
                    score -= closest_food_dist * 60 * food_weight_multiplier
                    score += (100 - my_health) * 8
                elif need_length and my_health < 60:
                    # Need length and health getting low
                    score -= closest_food_dist * 40 * food_weight_multiplier
                elif my_health > 70:
                    # Healthy - avoid food to focus on space control and aggression
                    score += closest_food_dist * 3

                # 1v1 FOOD DENIAL: control food when ahead
                if game_phase == "1v1" and len(opponent_heads) > 0:
                    opponent = opponent_heads[0]

                    if my_length > opponent["length"]:
                        # For each food, check if opponent is closer
                        for food_pos in food:
                            my_dist_to_food = manhattan_distance(next_pos, food_pos)
                            opp_dist_to_food = manhattan_distance(opponent["pos"], food_pos)

                            # If opponent is closer to this food, try to block it
                            if opp_dist_to_food < my_dist_to_food:
                                score -= my_dist_to_food * 25

                            # If we're closer and healthy, take it to maintain advantage
                            elif my_dist_to_food < opp_dist_to_food and my_health < 60:
                                score -= my_dist_to_food * 15

        # AGGRESSION - context-dependent based on advantages
        for opponent in opponent_heads:
            opponent_dist = manhattan_distance(next_pos, opponent["pos"])
            opponent_id = opponent.get("id", "")
            opponent_territory = voronoi_territories.get(opponent_id, 0)

            # Determine if we have territorial advantage
            territory_advantage = my_territory > opponent_territory

            # Late game and 1v1: strong aggression with advantages
            if game_phase in ["late", "1v1"]:
                # If we're longer OR have territory advantage, be aggressive
                if my_length > opponent["length"] or territory_advantage:
                    score -= opponent_dist * 5

                    # Extra aggressive in 1v1
                    if game_phase == "1v1":
                        score -= opponent_dist * 10

                    # Even more aggressive if we have both length AND territory advantage
                    if my_length > opponent["length"] and territory_advantage:
                        score -= opponent_dist * 8

                # If we're smaller AND losing territory, maintain distance
                elif my_length < opponent["length"] and not territory_advantage:
                    if opponent_dist < 3:
                        score += opponent_dist * 15

                # 1v1 FUTURE HEAD-TO-HEAD CONTROL: predict and block opponent moves
                if game_phase == "1v1":
                    # Predict opponent's possible moves
                    opponent_possible_moves = predict_opponent_moves(
                        opponent["pos"], opponent["length"], game_state, snake_positions
                    )

                    # Control spaces opponent wants to move to
                    for opp_dir, opp_move_data in opponent_possible_moves.items():
                        opp_next_pos = opp_move_data["pos"]
                        dist_to_their_move = manhattan_distance(next_pos, opp_next_pos)

                        if my_length > opponent["length"]:
                            # Get CLOSER to their options to intimidate/block
                            score -= dist_to_their_move * 30
                        else:
                            # Stay AWAY from their options
                            score += dist_to_their_move * 20

                    # Reward moves that limit opponent's good options
                    bad_options_count = sum(1 for move_data in opponent_possible_moves.values()
                                          if move_data["spaces"] < opponent["length"])
                    total_options = len(opponent_possible_moves)

                    if total_options > 0:
                        # If most of their moves are bad, we're in a great position
                        bad_ratio = bad_options_count / total_options
                        score += bad_ratio * 500

            # Early game: mild aggression if we have strong territory advantage
            elif game_phase == "early":
                # If we have huge territory advantage, push them a bit
                if territory_advantage and my_territory > opponent_territory * 1.5:
                    score -= opponent_dist * 2

                # If they're getting too close to our food/space, push back
                if opponent_dist < 4 and territory_advantage:
                    score -= opponent_dist * 1

        # CENTER CONTROL - territory dominance
        center_x = board_width // 2
        center_y = board_height // 2
        center_dist = abs(next_pos["x"] - center_x) + abs(next_pos["y"] - center_y)

        if game_phase == "early":
            # Early game: moderate center preference (food still priority)
            score -= center_dist * 5
        elif game_phase == "late":
            # Late game: strong center control for territory
            score -= center_dist * 10
        elif game_phase == "1v1":
            # 1v1: maximum center control
            score -= center_dist * 15

        # PREFER OPEN SPACE - bonus for moves with more escape routes
        # But only apply this when not aggressively seeking food
        if safe_neighbors >= 3 and not (game_phase == "early" and need_length):
            score += 50  # Good position with multiple exits

        move_scores[direction] = score

    # Get moves that won't immediately kill us (score > -10000)
    safe_moves = [d for d in directions if move_scores[d] > -10000]

    # Choose best available move
    next_move = max(safe_moves, key=lambda m: move_scores[m]) if safe_moves else max(directions, key=lambda m: move_scores[m])

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
