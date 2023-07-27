import json
import configparser
import datetime
import os


class WorkloadManager:

  def __init__(self, workload_json, states_fn):
    self.workload = workload_json
    self.states_fn = states_fn
    self.cur_timestamp = None
    self.cpu_params = None

  def get(self, key):
    if key in self.workload:
      return self.workload[key]
    return None
  def get_params(self):
    return self.workload["params"]

  def get_config(self):
    return self.workload["config"]
  
  def get_extra_config(self):
    return self.workload["extra_config"]

  def get_name(self):
    return self.workload["name"]

  def get_binaries(self):
    return self.workload["gem5_binaries"]

  def get_timestamp(self):

    if self.cur_timestamp:
      return self.cur_timestamp

    rn = datetime.datetime.now()
    cur_timestamp = f"{rn.hour}.{rn.minute}.{rn.second}-{rn.year}-{rn.month}-{rn.day}"

    self.cur_timestamp = cur_timestamp
    return cur_timestamp

  def load_config(self):

    if self.cpu_params:
      return self.cpu_params

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(os.path.join("../gem5_configs", self.get_config()))

    if self.get("extra_config"):
      config.read(os.path.join("../gem5_configs", self.get("extra_config")))

    cpu_params = []
    sections = config.sections()
    print(sections)
    for section in sections:
      for key in config[section]:
        print(key)
        t = {
          'opt_name' : key,
          'val' : config[section][key]
        }
        cpu_params.append(t)

    self.cpu_params = cpu_params
    return cpu_params


  def gen_stats_name(self, params):
    return self.states_fn(params)


        
