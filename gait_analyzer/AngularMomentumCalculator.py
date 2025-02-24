import biorbd
import numpy as np


class AngularMomentumCalculator:
    def __init__(self, biorbdmodel, q_filtered, qdot):
        self.biorbdmodel = biorbdmodel
        self.q = q_filtered
        self.qdot = qdot

        # Extraire les masses et matrices d'inertie depuis le modèle
        self.mass, self.inertias = self.extract_mass_inertias()

        # Calcul des positions et vitesses du centre de masse de chaque segment à toutes les frames
        self.com_positions, self.com_velocities = self.calculate_com_positions_velocities()

    def extract_mass_inertias(self):
        """ Récupère les masses et matrices d'inertie depuis le modèle bioMod """
        mass = []
        inertias = []
        for i in range(self.biorbdmodel.nbSegment()):
            segment = self.biorbdmodel.segment(i)
            mass.append(segment.characteristics().mass())
            inertias.append(segment.characteristics().inertia().to_array())  # Matrice d'inertie sous forme de numpy array
        return np.array(mass), np.array(inertias)

    def calculate_com_positions_velocities(self):
        """
        Calcule la position et la vitesse du centre de masse de chaque segment pour toutes les frames.
        """
        n_frames = self.q.shape[1]  # Nombre de frames
        n_segments = self.biorbdmodel.nbSegment()

        com_positions = np.zeros((n_segments, n_frames, 3))  # Stocke les positions des CoM
        com_velocities = np.zeros((n_segments, n_frames, 3))  # Stocke les vitesses des CoM

        for f in range(n_frames):
            for i in range(n_segments):
                com_positions[i, f, :] = self.biorbdmodel.CoMbySegment(self.q[:, f], i).to_array()
                com_velocities[i, f, :] = self.biorbdmodel.CoMdotBySegment(self.q[:, f], self.qdot[:, f], i).to_array()

        return com_positions, com_velocities

    # def angular_momentum_trunk(self, index_trunk):
    #     """
    #     Calcule le moment angulaire du tronc sur toutes les frames.
    #     """
    #     H_trunk = np.zeros((self.q.shape[1], 3))  # Stocke H pour toutes les frames
    #
    #     for f in range(self.q.shape[1]):
    #         r_trunk = self.com_positions[index_trunk, f]  # Position du centre de masse du tronc
    #         v_trunk = self.com_velocities[index_trunk, f]  # Vitesse du centre de masse du tronc
    #         m_trunk = self.mass[index_trunk]  # Masse du tronc
    #         I_trunk = self.inertias[index_trunk]  # Matrice d'inertie
    #         omega_trunk = self.qdot[:, f]  # Vitesse angulaire du tronc (à adapter si nécessaire)
    #
    #         # Calcul du moment angulaire
    #         H_trunk[f, :] = np.cross(r_trunk, m_trunk * v_trunk) + np.dot(I_trunk, omega_trunk)
    #
    #     return H_trunk

    def calculate_angular_momentum(self):
        """
        Calcule le moment angulaire total en 3D en appliquant la formule :
        H = Σ ( r_i ∧ (m_i * v_i) + I_i * ω_i )
        """
        n_segments = len(self.mass)
        n_frames = self.q.shape[1]
        H_total = np.zeros((n_frames, 3))  # Stocke H pour toutes les frames

        for f in range(n_frames):
            for i in range(n_segments):
                r_i = self.com_positions[i, f]  # Position du centre de masse du segment i
                v_i = self.com_velocities[i, f]  # Vitesse du centre de masse du segment i
                m_i = self.mass[i]  # Masse du segment i
                I_i = self.inertias[i]  # Matrice d'inertie du segment i
                omega_i = self.qdot[:, f]  # Vitesse angulaire du segment i (à adapter si besoin)

                # Calcul du moment angulaire pour le segment i
                H_i = np.cross(r_i, m_i * v_i) + np.dot(I_i, omega_i)

                # Ajouter au moment angulaire total
                H_total[f, :] += H_i

        return H_total

    def outputs(self):
        return {
            "whole body angular momentum": self.calculate_angular_momentum(),
            # "angular momentum of trunk": self.angular_momentum_trunk(index_trunk),
        }
