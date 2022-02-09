The intent is to decouple serving service and serving engine instances but allow some form of integration across. 
Need to scale Triton with k8s and also cater for individual triton deployment but all should connect / able to connect to serving service to have aggregated view. 


![image](https://user-images.githubusercontent.com/55354225/153176658-311afedb-641a-4aca-919f-98e7662bd8f7.png)


**Some more work required **

1. to reset state in serving service (if triton reset**).  Right now,  previous info on triton metrics are not removed.

2. handle cases where there are >1 triton replicas. Right now, the replica set works, and both can connect to serving service, report metrics. Just that the metrics presentation in serving service is not handled for >1 triton.

3. check why serving state not updated when more models are published. Might be bug, check with clearml-serving.

4. current clearml-serving unload and load model regardless if the state has changed. if there are many models, it may affect performance. To confirm this and update codes to only reload if state has changed over at serving service.

5. Find a means to pass in serving-id entrypoint, to minimise changing triton image. Right now, serving-id is hardcoded in the script inside triton image.

** Side topic: Triton does not preserve the state once restart/reset. Maybe it will be good to preserve it for resource / usage monitoring. To check with Triton. 
