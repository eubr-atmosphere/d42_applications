# Sample applications for the ATMOSPHERE Elastic Kubernetes

## Integrated folder

The sample includes a sshserver (05) that exposes a persistent volume (01 and 02). The variables with the access keys are included in a secret K8s object (04) and the service address of the server is registered in a config map (03). The application has different posibilities:
* A Jupyter notebook as a deployment (06) which mounts the the sshfolder and a deployment with a cluster of workers which also mount the filesystem (07). The integrated version is `integrated.yaml`.
* A Jupyter notebook as a front-end of a distributed Tensorflow with a parameter server statefulset (08) and a statefulset of worker nodes (09). The configuration of both statefulsets is defined in a ConfigMap (`distribtf_config.yaml`). The integrated version is in `integrated_gpu_tf3.yaml`
* An MPI cluster that can be run from the Jupyter Notebook. In this case, the hostfile is defined through a configMap. The integrated version is in `integratedgpumpi.yaml`.
