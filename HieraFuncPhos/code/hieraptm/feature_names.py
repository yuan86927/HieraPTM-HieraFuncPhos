"""Numerical components extracted from the frozen final Task2 snapshot."""
from __future__ import annotations
MOTION = ["signed_gnm_mean","gnm_std","sign_agreement","anm_directional","anm_available",
          "center_log_flex","partner_log_flex","log_relative_flex","distance_100a","log_sequence_distance"]
RESPONSE = ["log_prs_center_to_partner","log_prs_partner_to_center","center_log_relative_dfi",
            "partner_log_relative_dfi","center_log_self_response","partner_log_self_response",
            "center_log_effector_potential","partner_log_effector_potential"]
NETWORK = ["log_weighted_degree","participation","external_strength_fraction","community_fraction",
           "within_community_strength_z","remote_strength_fraction","log_scaled_pagerank",
           "positive_strength_fraction","negative_strength_fraction","same_community_as_center"]
FEATURE_NAMES = MOTION + RESPONSE + NETWORK
