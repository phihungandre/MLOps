from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):
    @task
    def generate_text(self):
        self.client.post("/generate",
                         json={"prompt": "Votre texte de départ ici"},
                         headers={"Authorization": "your_token"})

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)

# Run the Locust load test with the following command: locust -f locustfile.py --host=http://<EXTERNAL-IP>