# camera-trap-species-identifier-v1.1-
Example 1
--------------------------------
True Label      : gallina
Predicted Label : gallina
Confidence      : 0.785
Uncertainty     : 0.180
Review Category : REVIEW: LOW CONFIDENCE

Reason:
Although the prediction is correct, the uncertainty is high, so the image is sent for human review.

Example 2
--------------------------------
True Label      : cavallo
Predicted Label : cavallo
Confidence      : 0.996
Uncertainty     : 0.006
Review Category : HIGH CONFIDENCE

Reason:
The model is highly confident and uncertainty is extremely low, so the image is automatically filed.

Example 3
--------------------------------
True Label      : ragno
Predicted Label : farfalla
Confidence      : 0.836
Uncertainty     : 0.181
Review Category : REVIEW: LOW CONFIDENCE

Reason:
The model made an incorrect prediction and the high uncertainty correctly triggered human review.

Example 4
--------------------------------
True Label      : ragno
Predicted Label : ragno
Confidence      : 0.991
Uncertainty     : 0.012
Review Category : HIGH CONFIDENCE

Reason:
The model is very certain about the prediction.

Example 5
--------------------------------
True Label      : gallina
Predicted Label : gallina
Confidence      : 0.854
Uncertainty     : 0.164
Review Category : REVIEW: LOW CONFIDENCE

Reason:
Confidence is acceptable, but uncertainty exceeds the threshold, therefore the image is flagged for review.


The system reduced manual workload from 100% to only 50% requiring immediate human review, with an additional 30% flagged for uncertain cases and 20% prioritized as potential rare species sightings.