from jina import Flow


f = (
    Flow(protocol="HTTP", port=5124)
    .add(name="detector", uses="docker://detecting-matching-detection")
    .add(name="matcher", uses="docker://detecting-matching-similarity-search", needs="detector")
)
# f.to_docker_compose_yaml("a.yml")  # noqa: ERA001
# f.plot("a.png")  # noqa: ERA001

with f:
    f.block()
