import numpy as np

class AngularMomentumCalculator:
    def __init__(self, biorbd_model, q_filtered, qdot, subject_mass, subject_height):
        self.biorbd_model = biorbd_model  # Ajout du modèle Biorbd
        self.q = q_filtered
        self.qdot = qdot
        self.subject_mass = subject_mass
        self.subject_height = subject_height
        self.H_total = None  # Initialisation du moment cinétique total
        self.H_total_normalized = None  # Initialisation du moment cinétique normalisé

    def calculate_angular_momentum(self):
        """
        Calcule le moment cinétique total en 3D pour toutes les frames.
        """
        n_frames = self.q.shape[1]  # Nombre de frames

        # Initialisation de H_total (matrice de taille n_frames x 3)
        self.H_total = np.zeros((n_frames, 3))

        for f in range(n_frames):
            H_f = np.array([
                m.to_array() for m in self.biorbd_model.CalcSegmentsAngularMomentum(self.q[:, f], self.qdot[:, f], True)
            ])
            self.H_total[f, :] += np.sum(H_f, axis=0)

        return self.H_total

    def normalize_angular_momentum(self):
        """
        Normalise le moment cinétique total par (Masse totale * Hauteur * sqrt(g * Hauteur)) pour chaque frame.
        """
        if self.H_total is None:  # Vérifie si H_total a déjà été calculé
            self.calculate_angular_momentum()

        g = 9.80665  # Accélération gravitationnelle en m/s²
        normalization_factor = self.subject_mass * self.subject_height * np.sqrt(g * self.subject_height)

        # Normalisation sur toutes les frames
        self.H_total_normalized = self.H_total / normalization_factor
        return self.H_total_normalized

    def outputs(self):
        return {
            "whole body angular momentum": self.H_total_normalized,
        }
