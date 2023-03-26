from significance_policies.BaseSigPolicy import SigPolicy
import json

class TempPolicy(SigPolicy):
    def __init__(self, logger):
        super().__init__()
        self._name = "TempPolicy"

        self.significance_trace_path = "/home/netlab/DL_lab/opacus_simulation/traces/significance_{}.json".format(self.name)
        self.logger = logger
        with open(self.significance_trace_path, "r+") as f:
            self.significance_trace = json.load(f)

    def get_job_datablock_signficance(self, signficance_state):
        target_dataset_name = signficance_state["target_dataset_name"]
        train_type = signficance_state["train_type"]
        test_type = signficance_state["test_type"]
        epsilon_consume = signficance_state["epsilon_consume"]
        return self.significance_trace[target_dataset_name]["sub_train_{}".format(train_type)]["sub_test_{}".format(test_type)]["epsilon_consume_{}".format(epsilon_consume)]