import argparse
import os
from tqdm import tqdm
from pipeline import run_pipeline
from core.steganography import SteganographyCore
# 这里可以从args读取 prompt 文件，做循环，为了方便，你可以等骨架跑通了再写