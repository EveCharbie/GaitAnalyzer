from collections import defaultdict
import numpy as np
import biorbd

class AngularMomentumCalculator:
    def __init__(self, biorbd_model, q_filtered, qdot, subject_mass, subject_height, subject_names):
        self.model = biorbd_model
        self.q = q_filtered
        self.qdot = qdot
        self.subject_name = subject_names
        self.subject_mass = subject_mass
        self.subject_height = subject_height

        self.H_total = None
        self.H_total_normalized = None
        self.H_seg = None
        self.segments_to_keep = None

    def calculate_angular_momentum(self):

        n_frames = self.q.shape[1]

        self.H_total = np.zeros((3, n_frames))
        self.H_seg = np.zeros((self.model.nbSegment(), 3, n_frames))

        # Récupérer tous les noms de DoF
        dof_names = [self.model.nameDof()[i].to_string() for i in range(self.model.nbDof())]

        # Extraire la dernière DoF avant changement de segment
        def extract_last_dof_per_segment(dof_names):
            last_dof_lines = []
            current_segment = None

            for i, dof_name in enumerate(dof_names):
                if "_translation" in dof_name:
                    segment_name = dof_name.split("_translation")[0]
                elif "_rotation" in dof_name:
                    segment_name = dof_name.split("_rotation")[0]
                else:
                    segment_name = dof_name.split("_")[0]

                if current_segment is not None and segment_name != current_segment:
                    last_dof_lines.append(dof_names[i - 1])
                current_segment = segment_name

            if dof_names:
                last_dof_lines.append(dof_names[-1])

            return last_dof_lines

        # Extraire les derniers DoFs par segment
        last_dof_names = extract_last_dof_per_segment(dof_names)

        # Trouver les indices des segments correspondants aux derniers DoFs extraits
        segments_to_keep = set()
        for dof_name in last_dof_names:
            # Extraire le nom du segment avant _translation ou _rotation
            if "_translation" in dof_name:
                segment_name = dof_name.split("_translation")[0]
            elif "_rotation" in dof_name:
                segment_name = dof_name.split("_rotation")[0]
            else:
                segment_name = dof_name.split("_")[0]

            # Chercher l'indice du segment dans le modèle
            for j in range(self.model.nbSegment()):
                segment_name_model = self.model.segment(j).name().to_string()
                if segment_name_model == segment_name:
                    segments_to_keep.add(j)
                    break

        self.segments_to_keep = sorted(segments_to_keep)

        # Dictionnaire de correspondance : segment anatomique → indices dans seg_H
        segment_index_map = defaultdict(list)

        # On calcule ça une seule fois (pas besoin à chaque frame)
        # On suppose que seg_H retourne autant d’éléments qu’il y a de DoFs
        q_0 = self.q[:, 0]
        qdot_0 = self.qdot[:, 0]
        seg_H_0 = self.model.CalcSegmentsAngularMomentum(q_0, qdot_0, True)

        for j in range(len(seg_H_0)):
            # `seg_H_0[j]` est associé à un segment
            segment = self.model.segment(j).name().to_string()

            # Retrouver l'indice de ce segment dans la liste officielle
            for k in range(self.model.nbSegment()):
                if self.model.segment(k).name().to_string() == segment:
                    segment_index_map[k].append(j)
                    break

        # Boucle principale
        for i in range(n_frames):
            q_i = self.q[:, i]
            qdot_i = self.qdot[:, i]

            self.H_total[:, i] = self.model.angularMomentum(q_i, qdot_i, True).to_array()
            seg_H = self.model.CalcSegmentsAngularMomentum(q_i, qdot_i, True)

            for k in range(self.model.nbSegment()):
                for j in segment_index_map[k]:
                    self.H_seg[k, :, i] += seg_H[j].to_array()

        segments_to_keep = sorted(list(segments_to_keep))  # au cas où ce serait un set
        self.H_seg = self.H_seg[segments_to_keep]
        return self.H_total, self.H_seg

    def outputs(self):
        return {
            "segment_angular_momentum": self.H_total,
            "order_segment": self.H_seg
        }
