ZEBRA_GRID_ORIGINAL = """
# Example Puzzle 

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

Reasoning: Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.

{
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve 

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following format:

Reasoning: ...

```json
{json_template}
```

"""

ZEBRA_GRID_EGOR = """

THE PUZZLE: 
{puzzle}

INSTRUCTIONS:
Solve the puzzle independently by following all clues carefully. In the beginning describe semantics of all relations like: in between, left, right next, etc - in a formal logic way using math expressions. After this describe all clues and conditions in a formal logic way with math expressions based on the relations. Proceed step by step using formal logic expressions, showing your reasoning at each stage. For every step: Represent the current state clearly in a table format for each house - number, name, education, color, nationality, birth_month.  Do not alter any positions or assignments that have already been confirmed as definitely correct. Include all relevant constraints and known facts before applying that step. Verify that the table satisfies the constraints up to that point. Keep records about conclusions that you found definitely wrong. Continue reasoning thoroughly until you reach a final consistent solution. After completing, double-check the entire solution against all constraints to ensure accuracy. When the final answer is found and verified provide it in full accordance with the template.

FINAL ANSWER TEMPLATE:
```json
{json_template}
```

"""

ZEBRA_GRID = ZEBRA_GRID_ORIGINAL