class Metrics:
    def __init__(self, metrics: dict = None):
        self.loss = metrics["loss"] if metrics else []
        self.accuracy = metrics["accuracy"] if metrics else []
        self.r_squared = metrics["r_squared"] if metrics else []

    def __str__(self):

        loss = self.loss[-1] if self.loss else "N/A"
        accuracy = self.accuracy[-1] if self.accuracy else "N/A"
        r_squared = self.r_squared[-1] if self.r_squared else "N/A"

        return f"Loss: {loss}, Accuracy: {accuracy}, R^2: {r_squared}"

    def __repr__(self):
        return self.__str__()
    
    def as_dict(self):
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "r_squared": self.r_squared
        }
    
    def add_metric(self, loss: float, accuracy: float, r_squared: float):
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.r_squared.append(r_squared)