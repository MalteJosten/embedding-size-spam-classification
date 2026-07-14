import numpy as np

class Classifier():
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.mail_id = None
        self.best_val = float("inf")
        self.worst_val = float("-inf")
        self.best_id = None
        self.worst_id = None
        self.avg = None
        self.med = None
        self.values = []
        self.dists = {}
        self.orig_dist = None
        self.orig_in_top_5 = None
        self.orig_in_top_10 = None

    def set_mail_id(self, mail_id):
        self.mail_id = mail_id

    def compare(self, v1, k2, v2) -> float:
        return float("inf")

    def is_better_than_threshold(self, v, t) -> bool:
        return False

    def is_spam(self, input, knowledge_base):
        return False
    
    def determine_top(self) -> tuple[int, int]:
        return (0, 0)

    def sort_distances(self, distances):
        distances.sort(reverse=True)
        return distances

    def get_summary(self):
        self.avg = float(np.mean(self.values))
        self.med = float(np.median(self.values))

        self.orig_in_top_5, self.orig_in_top_10 = self.determine_top()

        return self.best_val, self.best_id, self.worst_val, self.worst_id, self.orig_dist, self.orig_in_top_5, self.orig_in_top_10, self.avg, self.med

    def get_best_data(self):
        return {}

class Euclidean(Classifier):
    def __init__(self):
        self.reset()

    def reset(self):
        super().reset()

    def set_mail_id(self, mail_id):
        self.mail_id = mail_id

    def compare(self, v1, k2, v2):
        if len(v1) != len(v2):
            return float("inf")

        result = float(np.linalg.norm(np.array(v1) - np.array(v2)))

        if self.mail_id == k2:
            self.orig_dist = result

        if result < self.best_val:
            self.best_val = result
            self.best_id = k2
        
        if result > self.worst_val:
            self.worst_val = result
            self.worst_id = k2

        self.values.append(result)
        self.dists.setdefault(result, []).append(k2)

    def get_distance(self, v1, v2):
        if len(v1) != len(v2):
            return float("inf")

        result = float(np.linalg.norm(np.array(v1) - np.array(v2)))

        return result

    def is_better_than_threshold(self, v, t) -> bool:
        return v <= t

    def determine_top(self) -> tuple[int, int]:
        distances = list(self.dists.keys())
        distances.sort()

        for i in range (10):
            for m_id in self.dists[distances[i]]:
                if m_id == self.mail_id:
                    if i >= 5:
                        return 1, 1
                    else:
                        return 1, 0

        return 0, 0

    def sort_distances(self, distances):
        distances.sort()
        return distances

    def get_best_data(self):
        return {
            "id": self.best_id,
            "value": self.best_val
        }


class CosineSimilarity(Classifier):
    def __init__(self):
        self.reset()
        self.best_norm1 = None
        self.best_norm2 = None
        self.best_dot = None

    def reset(self):
        super().reset()
        self.best_val = float("-inf")
        self.worst_val = float("inf")

    def set_mail_id(self, mail_id):
        self.mail_id = mail_id

    def compare(self, v1, k2, v2):
        if len(v1) != len(v2):
            return float("inf")

        v1 = np.array(v1)
        v2 = np.array(v2)
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        result = float(dot / (norm1 * norm2))

        if self.mail_id == k2:
            self.orig_dist = result

        if result > self.best_val:
            self.best_val = result
            self.best_id = k2
            self.best_dot = dot
            self.best_norm1 = norm1
            self.best_norm2 = norm2
        
        if result < self.worst_val:
            self.worst_val = result
            self.worst_id = k2

        self.values.append(result)
        self.dists.setdefault(result, []).append(k2)

    def get_distance(self, v1, v2):
        if len(v1) != len(v2):
            return float("inf")

        v1 = np.array(v1)
        v2 = np.array(v2)
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        result = float(dot / (norm1 * norm2))

        return result

    def is_better_than_threshold(self, v, t) -> bool:
        return v >= t

    def determine_top(self) -> tuple[int, int]:
        distances = list(self.dists.keys())
        distances.sort(reverse=True)

        for i in range (10):
            for m_id in self.dists[distances[i]]:
                if m_id == self.mail_id:
                    if i >= 5:
                        return 1, 1
                    else:
                        return 1, 0

        return 0, 0

    def get_best_data(self):
        return {
            "id": self.best_id,
            "dot": self.best_dot,
            "norm1": self.best_norm1,
            "norm2": self.best_norm2,
            "value": self.best_val
        }


def get_classifier(name):
    if name == "euclidean":
        classifier = Euclidean()
    elif name == "cosine":
        classifier = CosineSimilarity()
    else:
        print("Specify classifier name")
        exit()

    return classifier