import inspect
from typing import Any, Dict


class PipelineManager:
    def __init__(self, deps: Dict[str, Any], classes: list[type]):
        self.deps = deps
        self.classes = classes

        self.pipelines = {}

    def build(self):
        for cls in self.classes:
            sig = inspect.signature(cls.__init__)
            kwargs = {k: self.deps[k] for k in sig.parameters if k in self.deps}
            self.pipelines[cls.__name__] = cls(**kwargs)

        return self.pipelines

    def run(self):
        for pipeline in self.pipelines.values():
            pipeline.start()
