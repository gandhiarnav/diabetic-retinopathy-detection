"""
predict.py
──────────
Clinical prediction logic for Diabetic Retinopathy detection.

Given a retinal image, outputs:
  - DR Grade (0–4) with confidence
  - Macular Edema Risk (0–2) with confidence
  - Clinical recommendation (when to see a doctor)
  - Urgency level (routine / soon / urgent / emergency)

Usage:
  from predict import predict_image, predict_batch

  result = predict_image('path/to/retina.jpg')
  print(result['recommendation'])
"""

import os
import sys
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torchvision import transforms
from torch.nn.functional import softmax

# ── Make src importable ───────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent))

from config import MODEL_DIR, RESULTS_DIR, IMG_SIZE
from model import ConvNextModel, MultimodalModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# Clinical knowledge base
# ─────────────────────────────────────────────────────────────────────────────

DR_INFO = {
    0: {
        'name'       : 'No Diabetic Retinopathy',
        'short'      : 'No DR',
        'description': 'No signs of diabetic retinopathy detected. '
                       'The retina appears healthy with no visible damage.',
        'urgency'    : 'routine',
        'color'      : '#2ecc71',
        'emoji'      : '✅',
    },
    1: {
        'name'       : 'Mild Non-Proliferative DR',
        'short'      : 'Mild NPDR',
        'description': 'Early stage. Small microaneurysms (tiny bulges in '
                       'blood vessels) are present. Vision is usually unaffected.',
        'urgency'    : 'routine',
        'color'      : '#f1c40f',
        'emoji'      : '⚠️',
    },
    2: {
        'name'       : 'Moderate Non-Proliferative DR',
        'short'      : 'Moderate NPDR',
        'description': 'Blood vessels are blocked and damaged. Fluid and blood '
                       'may leak into the retina. Risk of vision changes is increasing.',
        'urgency'    : 'soon',
        'color'      : '#e67e22',
        'emoji'      : '🟠',
    },
    3: {
        'name'       : 'Severe Non-Proliferative DR',
        'short'      : 'Severe NPDR',
        'description': 'Many blood vessels are blocked, depriving large areas '
                       'of the retina of blood. High risk of developing new '
                       'abnormal blood vessels.',
        'urgency'    : 'urgent',
        'color'      : '#e74c3c',
        'emoji'      : '🔴',
    },
    4: {
        'name'       : 'Proliferative Diabetic Retinopathy',
        'short'      : 'Proliferative DR',
        'description': 'Advanced stage. New fragile blood vessels grow on the '
                       'retina and can bleed. Serious risk of vision loss or '
                       'complete blindness without immediate treatment.',
        'urgency'    : 'emergency',
        'color'      : '#8e44ad',
        'emoji'      : '🚨',
    },
}

EDEMA_INFO = {
    0: {
        'name'       : 'No Macular Edema',
        'short'      : 'No Edema',
        'description': 'No swelling detected in the macula (central retina).',
        'color'      : '#2ecc71',
    },
    1: {
        'name'       : 'Moderate Macular Edema Risk',
        'short'      : 'Moderate Edema',
        'description': 'Some fluid accumulation near the macula. '
                       'Can affect central vision if untreated.',
        'color'      : '#e67e22',
    },
    2: {
        'name'       : 'Clinically Significant Macular Edema',
        'short'      : 'Significant Edema',
        'description': 'Significant swelling at the macula — the most important '
                       'part of the retina for sharp, central vision. '
                       'Requires prompt treatment.',
        'color'      : '#e74c3c',
    },
}

URGENCY_RECOMMENDATIONS = {
    'routine': {
        'action'    : 'Schedule a routine eye examination',
        'timeframe' : 'Within 12 months',
        'details'   : [
            'Continue regular diabetes management',
            'Maintain blood sugar within target range (HbA1c < 7%)',
            'Monitor blood pressure (target < 130/80 mmHg)',
            'Annual retinal screening is recommended for all diabetics',
        ],
    },
    'soon': {
        'action'    : 'See an ophthalmologist soon',
        'timeframe' : 'Within 3–6 months',
        'details'   : [
            'Do not delay — moderate DR can progress quickly',
            'Bring your recent blood sugar and HbA1c records',
            'Ask your doctor about laser treatment options',
            'Strict blood sugar control is critical at this stage',
            'Avoid activities that raise blood pressure suddenly',
        ],
    },
    'urgent': {
        'action'    : 'See an ophthalmologist urgently',
        'timeframe' : 'Within 1–4 weeks',
        'details'   : [
            'Severe DR has a high risk of progressing to vision loss',
            'Anti-VEGF injections or laser therapy may be needed',
            'Do NOT wait for symptoms — vision loss can be sudden',
            'Inform your diabetes care team immediately',
            'Strictly control blood sugar, blood pressure, and cholesterol',
        ],
    },
    'emergency': {
        'action'    : '🚨 Seek immediate medical attention',
        'timeframe' : 'As soon as possible — within days',
        'details'   : [
            'Proliferative DR is a medical emergency',
            'New blood vessels can bleed at any time causing sudden blindness',
            'Vitrectomy, laser photocoagulation, or anti-VEGF may be required',
            'Go to an eye emergency unit or call your ophthalmologist TODAY',
            'Do not drive if you notice any new floaters or vision changes',
            'This stage is treatable — but only with immediate intervention',
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Combined urgency (DR + Edema together)
# ─────────────────────────────────────────────────────────────────────────────

def get_combined_urgency(dr_grade: int, edema_grade: int) -> str:
    """
    Returns the highest urgency level considering both DR and Edema.
    Edema can escalate urgency — e.g. Grade 1 DR + Grade 2 Edema → 'soon'
    """
    urgency_rank = {'routine': 0, 'soon': 1, 'urgent': 2, 'emergency': 3}
    rank_urgency = {v: k for k, v in urgency_rank.items()}

    dr_urgency    = DR_INFO[dr_grade]['urgency']
    # Edema escalation: each edema grade adds +1 rank to urgency
    edema_boost   = edema_grade   # 0 = no boost, 1 = +1, 2 = +2

    dr_rank       = urgency_rank[dr_urgency]
    combined_rank = min(dr_rank + edema_boost, 3)   # cap at emergency

    return rank_urgency[combined_rank]


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

_model_cache = {}

def load_model(mode: str = 'convnext'):
    """Loads model once and caches it."""
    if mode in _model_cache:
        return _model_cache[mode]

    ModelClass = ConvNextModel if mode == 'convnext' else MultimodalModel
    model      = ModelClass().to(device)
    ckpt_path  = MODEL_DIR / f'dr_model_{mode}.pth'

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'Model not found: {ckpt_path}\n'
            f'Download from Kaggle and place in models/ folder.'
        )

    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.eval()
    _model_cache[mode] = model
    print(f'Model loaded: {mode} ({ckpt_path.name})')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_image(img_path: str | Path) -> torch.Tensor:
    """Loads and preprocesses a single retinal image."""
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    image  = torchvision.io.read_image(str(img_path)).float() / 255.0
    image  = resize(image)
    return image   # (3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_image(
    img_path : str | Path,
    mode     : str = 'convnext',
    verbose  : bool = True,
) -> dict:
    """
    Predicts DR grade and macular edema risk for a single retinal image.

    Args:
        img_path : path to retinal image (.jpg or .png)
        mode     : 'convnext' (recommended) or 'image_only'
        verbose  : if True, prints the clinical report to terminal

    Returns:
        dict with full prediction details and clinical recommendation
    """
    model = load_model(mode)
    image = load_image(img_path).unsqueeze(0).to(device)   # (1, 3, H, W)
    text_features = torch.zeros(1, 512).to(device)

    # ── Forward pass ──────────────────────────────────────────────────────────
    dr_logits, edema_logits = model(image, text_features)

    dr_probs    = softmax(dr_logits,    dim=1).cpu().numpy()[0]
    edema_probs = softmax(edema_logits, dim=1).cpu().numpy()[0]

    dr_grade    = int(np.argmax(dr_probs))
    edema_grade = int(np.argmax(edema_probs))
    dr_conf     = float(dr_probs[dr_grade])
    edema_conf  = float(edema_probs[edema_grade])

    # ── Clinical logic ────────────────────────────────────────────────────────
    urgency         = get_combined_urgency(dr_grade, edema_grade)
    urgency_info    = URGENCY_RECOMMENDATIONS[urgency]

    result = {
        # Raw predictions
        'dr_grade'        : dr_grade,
        'dr_confidence'   : dr_conf,
        'dr_probabilities': {i: float(p) for i, p in enumerate(dr_probs)},
        'edema_grade'     : edema_grade,
        'edema_confidence': edema_conf,
        'edema_probs'     : {i: float(p) for i, p in enumerate(edema_probs)},

        # Clinical info
        'dr_name'         : DR_INFO[dr_grade]['name'],
        'dr_description'  : DR_INFO[dr_grade]['description'],
        'edema_name'      : EDEMA_INFO[edema_grade]['name'],
        'edema_description': EDEMA_INFO[edema_grade]['description'],

        # Recommendation
        'urgency'         : urgency,
        'action'          : urgency_info['action'],
        'timeframe'       : urgency_info['timeframe'],
        'details'         : urgency_info['details'],

        # Metadata
        'image_path'      : str(img_path),
        'model_used'      : mode,
    }

    if verbose:
        print_clinical_report(result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

def print_clinical_report(result: dict):
    dr    = result['dr_grade']
    edema = result['edema_grade']
    sep   = '═' * 60
    sep2  = '─' * 60

    print(f'\n{sep}')
    print(f'  DIABETIC RETINOPATHY SCREENING REPORT')
    print(f'{sep}')
    print(f'  Image : {Path(result["image_path"]).name}')
    print(f'  Model : {result["model_used"]}')
    print(f'{sep2}')

    # DR result
    emoji = DR_INFO[dr]['emoji']
    print(f'\n  {emoji}  DR GRADE: {dr} — {result["dr_name"]}')
    print(f'      Confidence : {result["dr_confidence"]*100:.1f}%')
    print(f'      {result["dr_description"]}')

    # Probability breakdown
    print(f'\n  Grade probabilities:')
    for i, prob in result['dr_probabilities'].items():
        bar   = '█' * int(prob * 30)
        arrow = ' ← predicted' if i == dr else ''
        print(f'    Grade {i} ({DR_INFO[i]["short"]:<18}) '
              f'{prob*100:>5.1f}%  {bar}{arrow}')

    print(f'\n{sep2}')

    # Edema result
    print(f'\n  👁️  MACULAR EDEMA: {result["edema_name"]}')
    print(f'      Confidence : {result["edema_confidence"]*100:.1f}%')
    print(f'      {result["edema_description"]}')

    print(f'\n{sep2}')

    # Recommendation
    urgency_icons = {
        'routine'  : '🟢',
        'soon'     : '🟡',
        'urgent'   : '🔴',
        'emergency': '🚨',
    }
    icon = urgency_icons[result['urgency']]
    print(f'\n  {icon}  RECOMMENDATION [{result["urgency"].upper()}]')
    print(f'      Action    : {result["action"]}')
    print(f'      Timeframe : {result["timeframe"]}')
    print(f'\n  What to do:')
    for detail in result['details']:
        print(f'    • {detail}')

    print(f'\n{sep}')
    print(f'  ⚠️  This is an AI screening tool only.')
    print(f'     Always consult a qualified ophthalmologist.')
    print(f'{sep}\n')


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_prediction(result: dict, save_path: str = None):
    """
    Creates a visual prediction card showing:
    - The retinal image
    - DR grade probability bar chart
    - Edema probability bar chart
    - Recommendation panel
    """
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')

    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35,
                          left=0.05, right=0.95,
                          top=0.88, bottom=0.08)

    ax_img   = fig.add_subplot(gs[:, 0])
    ax_dr    = fig.add_subplot(gs[0, 1])
    ax_edema = fig.add_subplot(gs[0, 2])
    ax_rec   = fig.add_subplot(gs[1, 1:])

    text_color = 'white'

    # ── Title ─────────────────────────────────────────────────────────────────
    urgency_icons = {'routine':'🟢','soon':'🟡','urgent':'🔴','emergency':'🚨'}
    icon = urgency_icons[result['urgency']]
    fig.suptitle(
        f'DR Screening Report   {icon} {result["urgency"].upper()}',
        fontsize=16, fontweight='bold', color=text_color, y=0.96
    )

    # ── Retinal image ─────────────────────────────────────────────────────────
    try:
        import torchvision
        img = torchvision.io.read_image(result['image_path']).permute(1,2,0).numpy()
        ax_img.imshow(img)
    except Exception:
        ax_img.text(0.5, 0.5, 'Image\nNot Found',
                    ha='center', va='center', color='white',
                    transform=ax_img.transAxes, fontsize=14)
        ax_img.set_facecolor('#2d2d44')

    dr    = result['dr_grade']
    edema = result['edema_grade']
    ax_img.set_title(
        f'Grade {dr} — {DR_INFO[dr]["short"]}\n'
        f'Confidence: {result["dr_confidence"]*100:.1f}%',
        color=DR_INFO[dr]['color'], fontsize=11, fontweight='bold'
    )
    ax_img.axis('off')

    # ── DR probability bars ───────────────────────────────────────────────────
    dr_probs = [result['dr_probabilities'][i] * 100 for i in range(5)]
    bars = ax_dr.barh(
        GRADE_LABELS[::-1], dr_probs[::-1],
        color=[DR_INFO[i]['color'] for i in range(4, -1, -1)],
        edgecolor='none', height=0.6
    )
    ax_dr.set_xlim(0, 110)
    ax_dr.set_title('DR Grade Probabilities', color=text_color,
                    fontsize=11, fontweight='bold')
    ax_dr.set_facecolor('#2d2d44')
    ax_dr.tick_params(colors=text_color, labelsize=9)
    ax_dr.spines['bottom'].set_color('#555')
    ax_dr.spines['left'].set_color('#555')
    ax_dr.spines['top'].set_visible(False)
    ax_dr.spines['right'].set_visible(False)
    ax_dr.set_xlabel('Probability (%)', color=text_color, fontsize=9)
    for bar, val in zip(bars, dr_probs[::-1]):
        ax_dr.text(val + 1, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=8, color=text_color)

    # ── Edema probability bars ────────────────────────────────────────────────
    edema_labels = ['No Edema', 'Moderate', 'Significant']
    edema_probs  = [result['edema_probs'][i] * 100 for i in range(3)]
    edema_colors = ['#2ecc71', '#e67e22', '#e74c3c']
    bars2 = ax_edema.barh(
        edema_labels[::-1], edema_probs[::-1],
        color=edema_colors[::-1],
        edgecolor='none', height=0.5
    )
    ax_edema.set_xlim(0, 110)
    ax_edema.set_title('Macular Edema Risk', color=text_color,
                       fontsize=11, fontweight='bold')
    ax_edema.set_facecolor('#2d2d44')
    ax_edema.tick_params(colors=text_color, labelsize=9)
    ax_edema.spines['bottom'].set_color('#555')
    ax_edema.spines['left'].set_color('#555')
    ax_edema.spines['top'].set_visible(False)
    ax_edema.spines['right'].set_visible(False)
    ax_edema.set_xlabel('Probability (%)', color=text_color, fontsize=9)
    for bar, val in zip(bars2, edema_probs[::-1]):
        ax_edema.text(val + 1, bar.get_y() + bar.get_height()/2,
                      f'{val:.1f}%', va='center', fontsize=8, color=text_color)

    # ── Recommendation panel ──────────────────────────────────────────────────
    urgency_bg = {'routine':'#1a472a','soon':'#7d6608',
                  'urgent':'#6e2121','emergency':'#4a1040'}
    ax_rec.set_facecolor(urgency_bg[result['urgency']])
    ax_rec.axis('off')

    rec_text  = f"{icon}  {result['action']}\n"
    rec_text += f"⏱  Timeframe: {result['timeframe']}\n\n"
    for detail in result['details'][:4]:
        rec_text += f"  •  {detail}\n"

    ax_rec.text(0.02, 0.95, rec_text,
                transform=ax_rec.transAxes,
                fontsize=9.5, color='white',
                verticalalignment='top',
                fontfamily='monospace',
                linespacing=1.6)
    ax_rec.set_title('Clinical Recommendation',
                     color=text_color, fontsize=11, fontweight='bold')

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.02,
             '⚠️  AI screening tool only — always consult a qualified ophthalmologist',
             ha='center', fontsize=9, color='#aaaaaa', style='italic')

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_path is None:
        img_name  = Path(result['image_path']).stem
        save_path = RESULTS_DIR / f'prediction_{img_name}.png'

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'Prediction card saved → {save_path}')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(
    img_paths : list,
    mode      : str = 'convnext',
    save_csv  : bool = True,
) -> list[dict]:
    """
    Runs prediction on a list of images and optionally saves results to CSV.

    Args:
        img_paths : list of image file paths
        mode      : model to use
        save_csv  : save results to results/batch_predictions.csv

    Returns:
        List of result dicts
    """
    results = []
    for path in img_paths:
        print(f'Processing: {Path(path).name}')
        result = predict_image(path, mode=mode, verbose=False)
        results.append(result)

    # Print summary
    print(f'\n{"="*50}')
    print(f'  BATCH PREDICTION SUMMARY ({len(results)} images)')
    print(f'{"="*50}')
    print(f'  {"Image":<25} {"DR Grade":<20} {"Conf":>6} {"Urgency"}')
    print(f'  {"─"*48}')
    for r in results:
        name = Path(r['image_path']).name[:24]
        print(f'  {name:<25} {r["dr_name"][:19]:<20} '
              f'{r["dr_confidence"]*100:>5.1f}%  {r["urgency"]}')

    if save_csv:
        rows = [{
            'image'           : Path(r['image_path']).name,
            'dr_grade'        : r['dr_grade'],
            'dr_name'         : r['dr_name'],
            'dr_confidence'   : f'{r["dr_confidence"]*100:.1f}%',
            'edema_grade'     : r['edema_grade'],
            'edema_name'      : r['edema_name'],
            'edema_confidence': f'{r["edema_confidence"]*100:.1f}%',
            'urgency'         : r['urgency'],
            'action'          : r['action'],
            'timeframe'       : r['timeframe'],
        } for r in results]

        import pandas as pd
        df = pd.DataFrame(rows)
        csv_path = RESULTS_DIR / 'batch_predictions.csv'
        df.to_csv(csv_path, index=False)
        print(f'\n  Results saved → {csv_path}')

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import glob

    # Find any test image
    test_images = (
        glob.glob('data/raw/idrid/train_images/IDRiD_0*.jpg')[:1] or
        glob.glob('data/raw/idrid/test_images/IDRiD_0*.jpg')[:1]
    )

    if not test_images:
        print('No test images found — place an image path in predict_image()')
    else:
        img_path = test_images[0]
        print(f'Testing with: {img_path}\n')

        result = predict_image(img_path, mode='convnext')
        visualise_prediction(result)
