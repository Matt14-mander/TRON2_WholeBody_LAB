"""Play the frozen S1 model_500.pt on a single TRON2 in MuJoCo."""
from run_sfyg_teacher import main


if __name__ == '__main__':
    main(student=True)
