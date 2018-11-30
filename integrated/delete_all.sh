#!/bin/bash
kubectl delete deployment clusterworker sshserver tf-notebook
kubectl delete svc sshserver tf-notebook
kubectl delete configmap sshfs-configmap
kubectl delete secret sshfs-secret
