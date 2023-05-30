class SigPolicy:

    def __init__(self):
        self._name = None

    @property
    def name(self):
        return self._name

    def get_job_datablock_significance_async(self, type_id, signficance_state, device_index):
        pass
    