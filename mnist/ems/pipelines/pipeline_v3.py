import ems

version_seed = "0001"

@ems.pipeline
def pipeline_flow(pipeline_name):
  """return a List of task"""
  gpu_instance = ems.ComputationInstance(vcpus=15, gpus=1, memory=15)

  flow = [
    ems.create_task(name="mnist", ver=pipeline_name, path="mnist/example", 
      computation=gpu_instance, deps={}),
  ]

  return flow
