class State:
    ready: bool
    canceling: bool
    job_name: str
    nb_jobs_in_queue: int
    steps_by_job: int
    current_job_in_queue: int
    current_job_step: int

    def __init__(self):
        self.set_ready()

    def get_status_message(self):
        if not self.steps_by_job:
            return self.job_name
        else:
            progress = int((self.current_job_step / self.steps_by_job) * 100)
            if self.nb_jobs_in_queue > 1:
                return f"{self.job_name} {self.current_job_in_queue + 1}/{self.nb_jobs_in_queue}: {progress}%"
            else:
                return f"{self.job_name}: {progress}%"

    def is_ready(self):
        return self.ready

    def is_canceling(self):
        return self.canceling

    def set_ready(self):
        self.ready = True
        self.canceling = False
        self.job_name = "Ready"
        self.nb_jobs_in_queue = 0
        self.steps_by_job = 0
        self.current_job_in_queue = 0
        self.current_job_step = 0

    def set_canceling(self):
        if self.is_ready():
            return
        # TODO: Need a better solution for this.
        # If a cancel is fired when the Gradio gallery is refreshing (after a last job step)
        # set_ready() at the end of the inference loop will not be called again by Gradio, the app will lock.
        # This reset the app to to ready
        if (
            self.steps_by_job
            and self.current_job_step == self.steps_by_job - 1
        ):
            self.set_ready()
            return
        self.canceling = True
        self.job_name = "Canceling, please wait."
        self.nb_jobs_in_queue = 0
        self.steps_by_job = 0
        self.current_job_in_queue = 0
        self.current_job_step = 0

    def set_job(
        self,
        job_name: str,
        log_to_console: bool = True,
        current_job_in_queue: int = 0,
        nb_jobs: int = 1,
        steps_by_job: int = 0,
    ):
        if log_to_console:
            print(job_name)

        if self.is_canceling():
            return
        self.ready = False
        self.job_name = job_name
        self.nb_jobs_in_queue = nb_jobs
        self.steps_by_job = steps_by_job
        self.current_job_in_queue = current_job_in_queue
        self.current_job_step = 0

    def update_job_progress(self, current_job_step):
        self.current_job_step = current_job_step


app = State()
