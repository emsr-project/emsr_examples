import ems

version_seed = "0001"

@ems.pipeline
def pipeline_flow(pipeline_name):
  """return a List of task"""
  gpu_instance = ems.ComputationInstance(vcpus=15, gpus=1, memory=15)

  flow = [
    ems.create_task(name="mnist/train", ver=pipeline_name, path="mnist/train", 
      computation=gpu_instance, deps={}),
    ems.create_task(name="mnist/eval", ver=pipeline_name, path="mnist/eval", 
      computation=gpu_instance, deps={"train": ems.Dep("mnist/train"),}),
  ]

  return flow
