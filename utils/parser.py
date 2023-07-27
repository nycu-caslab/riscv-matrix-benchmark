import re


class Gem5StatsParser:


  def __init__(self, filename):

    self.begin_stats_pat = r"-+ Begin Simulation Statistics -+"


  