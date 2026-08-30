# TravelGym

A Gymnasium-compatible environment for travel planning preference elicitation simulation through reinforcement learning. The public interaction contract is: Search exposes the complete candidate list, Action gathers natural-language preference evidence, and the Actor implicitly compares the evidence before submitting one visible option ID with Answer.

## Features

- **Gymnasium Compatible**: Fully compliant with the Gymnasium environment interface
- **Travel Planning Simulation**: Agents help users plan trips by eliciting preferences
- **Public Control Contract**: Enforces search/action/answer ordering, aspect ownership, visibility and duplicate-call checks
- **Natural-Language Evidence**: User preference IDs and correctness labels remain private to the simulator
- **Search Integration**: Agents can search for travel options through the environment tool
- **Configurable**: Flexible configuration system for different travel scenarios
- **Logging**: Verbose mode for debugging and detailed monitoring

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Quick Start

Please first set up your OPENAI_API_KEY as environment variable.

```python
import travelgym
from travelgym import TravelEnv, get_default_config

# Create environment with default configuration
config = get_default_config()
config.verbose = True  # Enable detailed logging
env = TravelEnv(config)

# Reset to get initial travel scenario
observation, info = env.reset()
print(f"User: {observation['feedback']}")

# Search for the current aspect first. Search returns the complete list.
obs, reward, terminated, truncated, info = env.step("[search] Search for the requested hotel in Los Angeles.")
print(f"Search results: {obs['feedback']}")

# Ask for natural-language evidence; candidates are not filtered by the environment.
obs, reward, terminated, truncated, info = env.step("[action] Which hotel features matter most to you?")
print(f"User: {obs['feedback']}")

# Submit exactly one ID that appeared in Search.
obs, reward, terminated, truncated, info = env.step("[answer] H13")
print(f"Result: {obs['feedback']}")
print(f"Step reward (always zero): {reward}")
print(f"Terminal reward: {env.get_terminal_reward()}")

env.close()
```

## Action Format

The environment accepts actions in four specific formats:

### 1. Search: `[search] <query>`

Search once for the current aspect. A successful Search returns every candidate in `task["all_options"][aspect]`; the environment never shrinks this list.

### 2. Action: `[action] <message>`

Ask a focused natural-language question or otherwise gather evidence from the user. Action updates only the public conversation and simulator-side preference queue; it does not filter candidates.

### 3. Answer: `[answer] <option_id>`

Recommend specific travel options by ID:

- **Example**: `[answer] H13` - Recommend hotel option H13
- **Example**: `[answer] F5` - Recommend flight option F5
- **Example**: `[answer] A2` - Recommend apartment option A2
- **Requirement**: Submit exactly one option ID that appeared in the current Search result.

### 4. Episode Termination: `[finish]`

End the current episode:

- **Example**: `[finish]` - Terminate the episode
- **Purpose**: End the episode when travel planning is complete

## Configuration

### Basic Configuration

```python
from travelgym import TravelGymConfig

config = TravelGymConfig(
    max_steps=20,
    verbose=True,
    data_mode="random",
    reward_version="travelgym-terminal-v1",
    require_action_before_answer=False,
)

env = TravelEnv(config)
```

### Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `max_steps` | Maximum steps per episode | `20` | Any positive integer |
| `verbose` | Enable verbose logging | `False` | `True`/`False` |
| `data_mode` | Scenario selection mode | `"random"` | `"random"`, `"single"`, `"list"` |
| `data_source` | Specific scenario to use | `"random"` | Scenario key string or list |
| `reward_version` | Terminal reward contract | `travelgym-terminal-v1` | Fixed value |
| `require_action_before_answer` | Require an evidence turn before Answer | `False` | `True`/`False` |
| `search_failure_interval` | Simulate search errors every N calls | `5` | Any positive integer |
| `elicitation_interval` | Proactive preference reveal interval | `3` | Any positive integer |

## Gymnasium Registration

The environment is automatically registered with Gymnasium upon import:

```python
import gymnasium as gym
import travelgym

# Use the registered environment
env = gym.make('TravelGym-v0')
```

## Data Flow

The environment follows a standard reinforcement learning cycle:

1. **Reset**: Initializes a new travel scenario and returns the initial observation
2. **Step**: Processes a Search, Action or Answer call, evaluates the simulator, and returns public feedback
3. **Reward Calculation**: Returns zero at interaction steps and computes one private terminal score at episode end
4. **Termination**: Episode ends when every aspect is answered or `[finish]` is called
5. **Truncation**: Episode ends when `max_steps` is reached without completion

## Environment Behavior

### Travel Planning Process

1. **Scenario Initialization**: Each episode contains a travel planning scenario with hidden user preferences
2. **Search**: The Actor searches the current aspect and receives the complete candidate list
3. **Action Evidence**: The simulated user replies in natural language; no preference ID or filtered subset is exposed
4. **Implicit Selection**: The Actor compares the evidence with all candidates and submits one visible ID
5. **Success Condition**: Episode succeeds when all submitted IDs are correct, with best-ID quality reported separately

### Available Travel Aspects

The environment supports comprehensive travel planning across multiple categories:

- **Hotels (H)**: Room types, amenities, ratings, costs
- **Flights (F)**: Routes, airlines, layovers, costs
- **Apartments (A)**: Room configurations, capacity, amenities
- **Restaurants (R)**: Cuisine types, ratings, price levels
- **Rental Cars (C)**: Brands, models, features, costs

### Reward System

TravelGym uses terminal-only `travelgym-terminal-v1` scoring. The bounded score is:

```text
clip((3.00*correct_completion
    + 0.30*answer_quality
    + 0.20*legal_chain_rate
    + 0.15*hidden_preference_hit_rate
    + 0.05*efficiency
    - policy_penalty) / 3.70, -1, 1)
```

Interaction steps always return `0.0`; terminal diagnostics are available to the trainer/evaluator only. Candidate shrink, filter precision/recall/F1, preference IDs and correctness labels are not part of the Actor observation.

### User Simulation Features

- **Implicit Preference Revealing**: Users reveal preferences naturally through conversation
- **Proactive Elicitation**: Users may proactively share preferences if conversation goes off-topic
- **Realistic Responses**: GPT-4o simulates realistic user behavior and preferences
- **Preference Tracking**: System tracks which preferences have been elicited

## Example Actions

### Complete Action Examples

```python
# For each aspect: Search -> Action (optional under the soft default) -> Answer.
env.step("[search] Search the requested hotel in Los Angeles.")
env.step("[action] Which amenities and room features do you prefer?")
env.step("[answer] H13")  # Hotel recommendation
env.step("[search] Search the requested flight.")
env.step("[action] Do you have any airline or layover preferences?")
env.step("[answer] F5")   # Flight recommendation

# End episode
env.step("[finish]")
```

### Effective Travel Planning Strategy

```python
# Search first so the complete options are visible.
env.step("[search] Search the requested hotel in Los Angeles.")

# Ask focused questions and compare the reply with the complete list yourself.
env.step("[action] Which hotel features matter most to you?")
env.step("[answer] H13")

# Or end if planning is complete
env.step("[finish]")
```

## API Requirements

### `.env` Configuration

TravelGym automatically loads the repository-root `.env` file when its
configuration is created.  Process environment variables still take
precedence, so CI/cluster launchers can override individual values.

Copy the project template and fill in one API key:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=<your-openai-compatible-api-key>
# or: DEEPSEEK_API_KEY=<your-deepseek-api-key>
OPENAI_BASE_URL=https://api.deepseek.com
USER_MODEL_NAME=deepseek-v4-flash
```

`OPENAI_API_KEY` takes precedence when both key variables are present.

### Getting API Keys

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)

### Setup Example

Do not commit `.env`; it is already covered by the repository `.gitignore`.
