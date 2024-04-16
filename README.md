# detecting-matching

This is a demo of how you can build an ml pipeline with isolated dockerized Jina microservices.  
Run it with
```sh
docker compose up
```
This docker compose is mostly generated automatically. You can generate it yourself by uncommenting in `main.py` and running
```sh
python main.py
```
You can also generate a k8s config by replacing `to_docker_compose_yaml` with `to_kubernetes_yaml` 
