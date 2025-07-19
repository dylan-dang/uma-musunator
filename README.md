# Uma Musunator

*Uma Musunator* is a Python-based automation tool designed for Windows that streamlines the 
process of rerolling accounts in the mobile game **Uma Musume: Pretty Derby**. Utilizing OpenCV 
for image recognition, it allows users to automatically reroll until a desired number of support 
cards that are fully configurable by the user are obtained, and stops preemptively if target is
impossible to be met.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dylan-dang/uma-musunator.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Edit the default configuration file with your desired setup and card targets and save
   it as `config.json`. If your uma musume exe cannot be found by the program, you can add
   the path to your config.

## Usage
Run the main script:
```bash
python main.py
```

## Possible Future Enhancements
- Add configurable banner selection (currently defaults to the 4th banner)
- Enable support for detecting and rerolling characters, not just support cards
- Implement automatic data linking and continued rolling after meeting target
- Improve UI change detection with more robust methods than pixel color matching
- Add support for multiple target configurations
- Add support for more pull currency income
