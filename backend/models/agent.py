"""
AI Agent Layer: Clinical Reasoning & Uncertainty Quantification
Post-processes model predictions into medical narratives
"""

from typing import List, Dict, Optional
import json


class ClinicalAgent:
    """
    AI Agent for clinical decision support.
    
    Functions:
    1. Generates uncertainty protocols (borderline cases)
    2. Contextualizes Grad-CAM attention into anatomical language
    3. Creates actionable clinical recommendations
    """
    
    def __init__(self):
        self.uncertainty_thresholds = {
            "high": (0.3, 0.7),     # Very close to decision boundary
            "moderate": (0.2, 0.8), # Somewhat uncertain
            "low": (0.0, 1.0),      # Confident
        }
        
        self.anatomical_regions = {
            "medial": ["medial femorotibial compartment", "medial meniscus"],
            "lateral": ["lateral femorotibial compartment", "lateral meniscus"],
            "patellofemoral": ["patellofemoral joint space", "patellar cartilage"],
            "osteophytes": ["osteophyte formation", "bone spur"],
            "joint_space": ["joint space narrowing", "cartilage loss"],
        }
        
        self.cost_ratios = {
            "false_negative": 10,  # High cost of missing OA (clinical risk)
            "false_positive": 1,   # Lower cost of over-referral (conservative)
        }
    
    def generate_uncertainty_protocol(
        self,
        stage1_prob: float,
        uncertainty_level: str,
        stage2_prob: Optional[float] = None,
    ) -> str:
        """
        Generate a disclaimer when probability is near decision boundary.
        
        This implements the "Uncertainty Protocol":
        If Stage 1 score is borderline (e.g., 55%), inform clinician
        of moderate confidence and recommend clinical correlation.
        """
        
        if uncertainty_level == "high":
            return (
                "⚠️ MODERATE CONFIDENCE: The model detects borderline features "
                "consistent with early joint space changes or subtle osteophyte formation. "
                "Diagnostic confidence is moderate; clinical correlation with patient pain levels, "
                "functional status, and physical exam findings is strongly advised. "
                "Consider serial imaging in 12-24 months for disease progression assessment."
            )
        
        elif uncertainty_level == "moderate":
            return (
                "ℹ️ STANDARD CONFIDENCE: The model confidence is within typical operating range. "
                "Findings are consistent with the indicated OA grade. "
                "Clinical correlation with symptoms is recommended."
            )
        
        else:  # low
            return (
                "✓ HIGH CONFIDENCE: The model has high confidence in this assessment. "
                "Radiographic findings are consistent with the indicated grade. "
                "Recommendation aligns with standard clinical practice."
            )
    
    def contextualize_prediction(
        self,
        stage1_prob: float,
        stage2_prob: Optional[float],
        attention_regions: List[Dict],
        traffic_light: str,
    ) -> str:
        """
        Translate raw probabilities + attention maps into clinical narrative.
        
        Uses Grad-CAM attention to explain WHERE the model looked.
        Combines with clinical decision tree to create actionable output.
        """
        
        stage1_confidence = 100 * (1 - abs(stage1_prob - 0.5) * 2)
        
        narrative_parts = []
        
        # Part 1: Risk stratification
        if traffic_light == "green":
            narrative_parts.append(
                f"**Risk Category: LOW RISK** (Stage 1 confidence: {stage1_confidence:.1f}%)\n"
                f"The radiograph does not show definite osteoarthritis features. "
                f"No immediate intervention is required. "
                f"Recommend standard preventive care and activity modification."
            )
        
        elif traffic_light == "yellow":
            narrative_parts.append(
                f"**Risk Category: MODERATE OA (KL-2)** (Confidence: {(1-abs(stage1_prob-0.5)*2)*100:.1f}%)\n"
                f"The radiograph shows definite joint space narrowing and early osteophyte formation. "
                f"This is consistent with Kellgren-Lawrence Grade 2 (mild) osteoarthritis. "
                f"**Recommend:** Conservative treatment including physical therapy, "
                f"NSAIDs, weight management, and activity modification."
            )
        
        else:  # red
            narrative_parts.append(
                f"**Risk Category: SEVERE OA (KL-3/4)** (Confidence: {(1-abs(stage1_prob-0.5)*2)*100:.1f}%)\n"
                f"The radiograph shows significant joint space narrowing and osteophyte formation. "
                f"This is consistent with Kellgren-Lawrence Grade 3-4 (moderate-severe) osteoarthritis. "
                f"**Recommend:** Orthopedic consultation for surgical planning "
                f"(joint replacement vs. arthroscopic intervention)."
            )
        
        # Part 2: Anatomical findings from Grad-CAM
        if attention_regions:
            top_regions = sorted(
                attention_regions,
                key=lambda x: x.get("intensity", 0),
                reverse=True
            )[:2]
            
            narrative_parts.append(
                f"\n**Anatomical Focus (Grad-CAM Attention):**"
            )
            
            for region in top_regions:
                region_name = region.get("region", "Unknown")
                intensity = region.get("intensity", 0.5)
                
                if intensity > 0.7:
                    focus_text = f"Strong attention on **{region_name}**"
                elif intensity > 0.4:
                    focus_text = f"Moderate attention on {region_name}"
                else:
                    focus_text = f"Mild attention on {region_name}"
                
                narrative_parts.append(f"- {focus_text} (intensity: {intensity:.2f})")
            
            # Add clinical interpretation
            if any("medial" in r.get("region", "").lower() for r in top_regions):
                narrative_parts.append(
                    "\n*Note: Medial compartment involvement is common in weight-bearing knee OA. "
                    "Recommend weight-bearing radiographs for serial follow-up.*"
                )
        
        # Part 3: Key recommendations
        narrative_parts.append(
            f"\n**Next Steps:**\n"
            f"1. Patient counseling on OA progression and treatment options\n"
            f"2. Physical therapy referral for joint preservation strategies\n"
            f"3. Serial imaging in {'6-12 months' if traffic_light == 'yellow' else 'immediate'} "
            f"to assess disease progression\n"
            f"4. Consider MRI if ligamentous injury suspected"
        )
        
        return "\n".join(narrative_parts)
    
    def generate_clinical_summary_for_pdf(
        self,
        patient_id: str,
        stage1_prob: float,
        stage2_prob: Optional[float],
        traffic_light: str,
    ) -> str:
        """
        Concise summary for PDF report (1 paragraph).
        """
        
        if traffic_light == "green":
            summary = (
                f"Patient {patient_id}: No definite osteoarthritis detected. "
                f"Radiographic features are within normal limits. "
                f"Recommendation: Standard preventive care."
            )
        elif traffic_light == "yellow":
            summary = (
                f"Patient {patient_id}: Kellgren-Lawrence Grade 2 (mild) osteoarthritis detected. "
                f"Joint space narrowing and early osteophyte formation noted. "
                f"Recommendation: Conservative treatment with physical therapy and symptom management."
            )
        else:  # red
            summary = (
                f"Patient {patient_id}: Kellgren-Lawrence Grade 3-4 (severe) osteoarthritis detected. "
                f"Significant joint space loss and osteophyte formation noted. "
                f"Recommendation: Orthopedic consultation for surgical intervention planning."
            )
        
        return summary
    
    def get_cost_analysis(self) -> Dict:
        """
        Return cost-benefit analysis of thresholds.
        Used in technical report to justify threshold selection.
        """
        return {
            "false_negative_cost": self.cost_ratios["false_negative"],
            "false_positive_cost": self.cost_ratios["false_positive"],
            "cost_ratio": f"{self.cost_ratios['false_negative']}:1 (FN:FP)",
            "justification": (
                "High cost of false negatives (missing progressive OA) justifies "
                "conservative threshold selection. Better to over-refer than under-diagnose."
            ),
        }
