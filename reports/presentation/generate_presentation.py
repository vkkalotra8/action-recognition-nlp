"""Generate the refined, publication-grade 10-slide academic presentation for the Action Recognition NLP project."""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "reports" / "presentation"
OUTPUT_PPTX = OUTPUT_DIR / "Action_Recognition_Final_Presentation.pptx"

# Project images
HEATMAP_PATH = PROJECT_ROOT / "results" / "plots" / "text_similarity_heatmap.png"
CONTACT_SHEET_PATH = PROJECT_ROOT / "results" / "predictions" / "playing_guitar_contact_test_contact_sheet.jpg"

# Color Palette (Dark Professional Academic Tech Theme)
COLOR_BG = RGBColor(11, 17, 32)          # #0B1120 Dark Space Navy
COLOR_CARD = RGBColor(30, 41, 59)        # #1E293B Slate 800
COLOR_CARD_BORDER = RGBColor(51, 65, 85)  # #334155 Slate 700
COLOR_CARD_ALT = RGBColor(20, 28, 45)    # #141C2D Slate Alt

COLOR_NODE_BG = RGBColor(15, 23, 42)     # #0F172A Node Box Fill
COLOR_NODE_BORDER = RGBColor(56, 189, 248) # #38BDF8 Node Border

COLOR_CYAN = RGBColor(6, 182, 212)        # #06B6D4 Cyan 500
COLOR_SKY = RGBColor(56, 189, 248)        # #38BDF8 Sky 400
COLOR_BLUE = RGBColor(59, 130, 246)       # #3B82F6 Blue 500
COLOR_INDIGO = RGBColor(99, 102, 241)     # #6366F1 Indigo 500
COLOR_GREEN = RGBColor(16, 185, 129)      # #10B981 Emerald 500
COLOR_RED = RGBColor(239, 68, 68)         # #EF4444 Red 500
COLOR_AMBER = RGBColor(245, 158, 11)      # #F59E0B Amber 500

COLOR_TEXT_WHITE = RGBColor(255, 255, 255)
COLOR_TEXT_LIGHT = RGBColor(226, 232, 240)   # #E2E8F0
COLOR_TEXT_MUTED = RGBColor(148, 163, 184)   # #94A3B8
COLOR_TEXT_CYAN = RGBColor(56, 189, 248)

FONT_HEADING = "Segoe UI"
FONT_BODY = "Segoe UI"


def set_slide_background(slide):
    """Fill slide with dark navy background."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG


def add_header(slide, title_text, category_text=""):
    """Standard header for content slides."""
    if category_text:
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = FONT_HEADING
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_CYAN

    title_top = Inches(0.68) if category_text else Inches(0.5)
    title_box = slide.shapes.add_textbox(Inches(0.8), title_top, Inches(11.7), Inches(0.65))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT_WHITE


def add_card(slide, left, top, width, height, bg_color=COLOR_CARD, border_color=COLOR_CARD_BORDER):
    """Add a rounded-rectangle card shape for modular content layout."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
    else:
        card.line.fill.background()
    return card


def add_flow_node(slide, left, top, width, height, title, subtitle="", bg_color=COLOR_NODE_BG, border_color=COLOR_SKY):
    """Add a diagram block/node."""
    node = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    node.fill.solid()
    node.fill.fore_color.rgb = bg_color
    node.line.color.rgb = border_color
    node.line.width = Pt(1)
    tf = node.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.05)
    p = tf.paragraphs[0]
    p.text = title
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_HEADING
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEXT_WHITE
    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.alignment = PP_ALIGN.CENTER
        p2.font.name = FONT_BODY
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = COLOR_TEXT_MUTED
    return node


def add_arrow_label(slide, left, top, width, height, arrow_text="──►"):
    """Add an arrow connector label."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = arrow_text
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN


def set_speaker_notes(slide, notes_text):
    """Set structured speaker notes on a slide."""
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = notes_text


def build_presentation():
    """Build the complete 10-slide presentation."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1)

    # Decorative top bar badge
    badge = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.8), Inches(4.5), Inches(0.4))
    badge.fill.solid()
    badge.fill.fore_color.rgb = RGBColor(15, 42, 74)
    badge.line.color.rgb = COLOR_CYAN
    badge.line.width = Pt(1)
    tf_badge = badge.text_frame
    tf_badge.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf_badge.paragraphs[0]
    p.text = "FINAL PROJECT PRESENTATION / VIVA"
    p.alignment = PP_ALIGN.CENTER
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN

    # Main Project Title
    title_box = slide1.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(1.6))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Semantic Prompt-Guided Human Action Recognition"
    p.font.name = FONT_HEADING
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEXT_WHITE

    p2 = tf.add_paragraph()
    p2.text = "using Vision-Language Fusion"
    p2.font.name = FONT_HEADING
    p2.font.size = Pt(30)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_SKY

    # Subtitle
    sub_box = slide1.shapes.add_textbox(Inches(0.8), Inches(3.1), Inches(11.7), Inches(0.6))
    tf_sub = sub_box.text_frame
    tf_sub.word_wrap = True
    p_sub = tf_sub.paragraphs[0]
    p_sub.text = "A Rigorous Study on Multimodal Feature Alignment, Label Leakage Prevention, and Explainable AI"
    p_sub.font.name = FONT_BODY
    p_sub.font.size = Pt(15)
    p_sub.font.color.rgb = COLOR_TEXT_MUTED

    # 4 Metadata Info Cards
    card_w = Inches(2.75)
    card_h = Inches(2.5)
    card_gap = Inches(0.23)
    card_top = Inches(4.2)

    meta_items = [
        ("STUDENT CANDIDATE", "[Student Name]", "[Enrollment / Roll Number]"),
        ("PROJECT SUPERVISOR", "[Guide / Supervisor Name]", "[Designation / Department]"),
        ("DEPARTMENT & COLLEGE", "[Department Name]", "[Institute / University Name]"),
        ("ACADEMIC SESSION", "Academic Year: 2025–2026", "Machine Learning Project Defense")
    ]

    for i, (label, val1, val2) in enumerate(meta_items):
        left = Inches(0.8) + i * (card_w + card_gap)
        add_card(slide1, left, card_top, card_w, card_h, bg_color=COLOR_CARD, border_color=COLOR_CARD_BORDER)

        tb = slide1.shapes.add_textbox(left + Inches(0.2), card_top + Inches(0.25), card_w - Inches(0.4), card_h - Inches(0.5))
        tf_c = tb.text_frame
        tf_c.word_wrap = True
        
        p_hdr = tf_c.paragraphs[0]
        p_hdr.text = label
        p_hdr.font.name = FONT_HEADING
        p_hdr.font.size = Pt(10)
        p_hdr.font.bold = True
        p_hdr.font.color.rgb = COLOR_CYAN

        p1 = tf_c.add_paragraph()
        p1.text = val1
        p1.font.name = FONT_HEADING
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_TEXT_WHITE
        p1.space_before = Pt(8)

        p2 = tf_c.add_paragraph()
        p2.text = val2
        p2.font.name = FONT_BODY
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLOR_TEXT_MUTED
        p2.space_before = Pt(6)

    set_speaker_notes(slide1, 
        "SPEAKER NOTES (Slide 1 — Title & Introduction):\n"
        "• WHAT TO SAY: Welcome the esteemed members of the examination committee and project guides. My name is [Student Name], and today I am presenting our final project titled 'Semantic Prompt-Guided Human Action Recognition using Vision-Language Fusion'.\n"
        "• TECHNICAL CONTEXT: This research investigates bridging computer vision (video frames) with natural language processing (action sentence embeddings) to transform traditional black-box action classification into an interpretable, semantic matching system.\n"
        "• KEY FOCUS: We will walk through the dataset engineering, our dual-branch system architecture, a critical finding regarding label leakage in multimodal fusion, quantitative ablation results, and our deployed explainable Streamlit application.\n"
        "• TRANSITION: Let us start with the core problem statement, research motivation, and the specific objectives we set out to achieve."
    )

    # =========================================================================
    # SLIDE 2: Problem, Motivation & Objectives
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "Problem Statement & Project Objectives", "Motivation & Background")

    col_w = Inches(5.6)
    left_x = Inches(0.8)
    top_y = Inches(1.5)

    # Left Column: Problem & Motivation
    add_card(slide2, left_x, top_y, col_w, Inches(5.3), bg_color=COLOR_CARD)
    tb = slide2.shapes.add_textbox(left_x + Inches(0.3), top_y + Inches(0.25), col_w - Inches(0.6), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "THE PROBLEM WITH TRADITIONAL HAR"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_RED

    p = tf.add_paragraph()
    p.text = "• Black-Box Predictions: Standard vision models assign discrete class indices (e.g. Class 4) without semantic awareness of what the action entails."
    p.font.name = FONT_BODY
    p.font.size = Pt(11.5)
    p.font.color.rgb = COLOR_TEXT_LIGHT
    p.space_before = Pt(8)

    p = tf.add_paragraph()
    p.text = "• No Conceptual Reasoning: Vision-only models ignore natural relationships between actions (e.g., PlayingGuitar and Drumming both involve rhythm and musical instruments)."
    p.font.name = FONT_BODY
    p.font.size = Pt(11.5)
    p.font.color.rgb = COLOR_TEXT_LIGHT
    p.space_before = Pt(6)

    p = tf.add_paragraph()
    p.text = "• Zero Decision Transparency: Users receive an opaque label with no confidence breakdown, ambiguity measure, or human-readable explanation."
    p.font.name = FONT_BODY
    p.font.size = Pt(11.5)
    p.font.color.rgb = COLOR_TEXT_LIGHT
    p.space_before = Pt(6)

    p = tf.add_paragraph()
    p.text = "THE VISION-LANGUAGE OPPORTUNITY"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.space_before = Pt(14)

    p = tf.add_paragraph()
    p.text = "• By projecting visual features into a continuous semantic text space, actions are classified via semantic proximity, enabling prompt guidance and transparent explainability."
    p.font.name = FONT_BODY
    p.font.size = Pt(11.5)
    p.font.color.rgb = COLOR_TEXT_LIGHT
    p.space_before = Pt(6)

    # Right Column: 4 Project Objectives
    right_x = Inches(6.8)
    add_card(slide2, right_x, top_y, col_w, Inches(5.3), bg_color=COLOR_CARD)
    tb_r = slide2.shapes.add_textbox(right_x + Inches(0.3), top_y + Inches(0.25), col_w - Inches(0.6), Inches(4.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "4 CORE PROJECT OBJECTIVES"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_SKY

    objectives = [
        ("1. Recognize Actions from Video Streams", "Accurately classify multi-frame human activities from raw video inputs across diverse domains (sports, music, daily tasks)."),
        ("2. Establish Strong Visual Baseline", "Extract 512-D frame features using pretrained ResNet18, aggregate via mean pooling, and benchmark with Logistic Regression."),
        ("3. Model Continuous Semantic Language Space", "Encode rich natural language descriptions into 384-D semantic embeddings using Sentence-BERT (all-MiniLM-L6-v2)."),
        ("4. Leakage-Safe Multimodal Fusion & XAI", "Develop a valid visual-to-semantic projection network, enforce strict label leakage prevention, and deploy an explainable Streamlit application.")
    ]

    for title, desc in objectives:
        p = tf_r.add_paragraph()
        p.text = title
        p.font.name = FONT_HEADING
        p.font.size = Pt(11.5)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(8)

        p_desc = tf_r.add_paragraph()
        p_desc.text = desc
        p_desc.font.name = FONT_BODY
        p_desc.font.size = Pt(10.5)
        p_desc.font.color.rgb = COLOR_TEXT_MUTED
        p_desc.space_before = Pt(2)

    set_speaker_notes(slide2,
        "SPEAKER NOTES (Slide 2 — Problem & Objectives):\n"
        "• WHAT TO SAY: In human action recognition, traditional computer vision pipelines classify video clips into integer labels like 'Class 4' or 'Drumming'. However, they function as black boxes with no conceptual understanding of the physical activity.\n"
        "• WHY IT MATTERS: In real-world applications—such as surveillance, healthcare monitoring, and sports analytics—users need to understand why an action was identified, what alternative actions looked similar, and how certain the model is.\n"
        "• OBJECTIVES: We structured our work around 4 clear goals: First, recognizing actions from raw video; Second, establishing a rigorous 512-D ResNet18 visual baseline; Third, encoding action descriptions into a 384-D language space with Sentence-BERT; and Fourth, building a leakage-safe fusion projection model with explainable AI.\n"
        "• TRANSITION: To evaluate this rigorously, we built our dataset foundation on a curated 9-class benchmark from UCF101."
    )

    # =========================================================================
    # SLIDE 3: Dataset & Preprocessing Pipeline
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "Dataset & Data Engineering Pipeline", "Data Foundation")

    # 4 Stat Cards Across Top
    stat_w = Inches(2.75)
    stat_h = Inches(1.3)
    stat_gap = Inches(0.23)
    stat_top = Inches(1.45)

    stats = [
        ("1,261", "TOTAL VIDEOS", "867 Train / 197 Val / 197 Test"),
        ("9", "ACTION CLASSES", "Balanced Activity Categories"),
        ("16", "FRAMES / VIDEO", "224 × 224 RGB Uniform Stride"),
        ("20,176", "PROCESSED FRAMES", "0 Corrupted / 0 Dropped")
    ]

    for i, (num, label, sub) in enumerate(stats):
        left = Inches(0.8) + i * (stat_w + stat_gap)
        add_card(slide3, left, stat_top, stat_w, stat_h, bg_color=COLOR_CARD)
        tb = slide3.shapes.add_textbox(left, stat_top + Inches(0.12), stat_w, stat_h - Inches(0.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = num
        p.alignment = PP_ALIGN.CENTER
        p.font.name = FONT_HEADING
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = COLOR_CYAN

        p2 = tf.add_paragraph()
        p2.text = label
        p2.alignment = PP_ALIGN.CENTER
        p2.font.name = FONT_HEADING
        p2.font.size = Pt(10)
        p2.font.bold = True
        p2.font.color.rgb = COLOR_TEXT_WHITE

        p3 = tf.add_paragraph()
        p3.text = sub
        p3.alignment = PP_ALIGN.CENTER
        p3.font.name = FONT_BODY
        p3.font.size = Pt(9)
        p3.font.color.rgb = COLOR_TEXT_MUTED

    # Middle: Action Classes Grid (9 distinct pill badges)
    mid_top = Inches(2.95)
    add_card(slide3, Inches(0.8), mid_top, Inches(11.7), Inches(1.35), bg_color=COLOR_CARD)
    tb_cls = slide3.shapes.add_textbox(Inches(1.0), mid_top + Inches(0.1), Inches(11.3), Inches(0.3))
    tf_cls = tb_cls.text_frame
    tf_cls.word_wrap = True
    p = tf_cls.paragraphs[0]
    p.text = "CURATED 9 ACTION CLASSES (UCF101 BENCHMARK SUBSET)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_SKY

    class_names = [
        "Archery", "Basketball", "Biking", "Bowling", "Drumming",
        "JavelinThrow", "PlayingGuitar", "RopeClimbing", "Typing"
    ]
    pill_w = Inches(1.18)
    pill_gap = Inches(0.08)
    pill_top = mid_top + Inches(0.48)

    for i, cname in enumerate(class_names):
        p_left = Inches(1.0) + i * (pill_w + pill_gap)
        add_flow_node(slide3, p_left, pill_top, pill_w, Inches(0.65), cname, "UCF101", bg_color=COLOR_NODE_BG, border_color=COLOR_SKY)

    # Bottom: Preprocessing Flowchart Steps
    flow_top = Inches(4.5)
    flow_w = Inches(2.75)
    flow_h = Inches(2.35)

    flow_steps = [
        ("Step 1: Ingestion", "Raw UCF101 Videos", "Loads .avi video clips containing variable durations and framerates across 9 distinct activity folders."),
        ("Step 2: Sampling", "16 Uniform Frames", "Evenly samples 16 frames across video duration; resizes and normalizes to 224 × 224 pixels (ImageNet RGB standard)."),
        ("Step 3: Partitioning", "Group-Safe Split", "Strictly partitions videos by recording group (gXX). No actor or background environment appears in both train and test!"),
        ("Step 4: Serialization", "Extracted Frame Tensors", "Saves verified frame directories and split metadata manifests (video_splits.csv), guaranteeing 100% reproducibility.")
    ]

    for i, (step_num, step_title, step_desc) in enumerate(flow_steps):
        left = Inches(0.8) + i * (flow_w + stat_gap)
        add_card(slide3, left, flow_top, flow_w, flow_h, bg_color=COLOR_CARD)
        
        tb = slide3.shapes.add_textbox(left + Inches(0.15), flow_top + Inches(0.15), flow_w - Inches(0.3), flow_h - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = step_num.upper()
        p.font.name = FONT_HEADING
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLOR_CYAN

        p2 = tf.add_paragraph()
        p2.text = step_title
        p2.font.name = FONT_HEADING
        p2.font.size = Pt(12)
        p2.font.bold = True
        p2.font.color.rgb = COLOR_TEXT_WHITE
        p2.space_before = Pt(2)

        p3 = tf.add_paragraph()
        p3.text = step_desc
        p3.font.name = FONT_BODY
        p3.font.size = Pt(10)
        p3.font.color.rgb = COLOR_TEXT_MUTED
        p3.space_before = Pt(6)

    set_speaker_notes(slide3,
        "SPEAKER NOTES (Slide 3 — Dataset & Preprocessing):\n"
        "• DATASET SUMMARY: We selected 9 action classes from the well-established UCF101 dataset, encompassing 1,261 total video clips.\n"
        "• FRAME SAMPLING: For each video, 16 frames were uniformly extracted across the entire time duration and resized to 224x224 RGB, generating exactly 20,176 frames with zero corruptions or unreadable files.\n"
        "• CRITICAL METHODOLOGY (GROUP-SAFE SPLITTING): A key engineering decision was enforcing Group-Safe partitioning (70% train, 15% val, 15% test). In UCF101, video clips share group IDs like 'g01' or 'g02' which represent the same actor, camera, and background room. By ensuring that all clips from group gXX stay in the same split, we eliminate subject/background leakage between train and test partitions.\n"
        "• TRANSITION: With this clean data foundation, let us examine the complete system architecture and its dual-branch design."
    )

    # =========================================================================
    # SLIDE 4: Complete System Architecture (Visually Structured Diagram)
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "Complete System Architecture", "System Design")

    # Section 1: Visual Branch Box
    add_card(slide4, Inches(0.8), Inches(1.45), Inches(11.7), Inches(1.75), bg_color=COLOR_CARD)
    tb_vh = slide4.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(0.3))
    tf_vh = tb_vh.text_frame
    p = tf_vh.paragraphs[0]
    p.text = "VISUAL ENCODER BRANCH (VIDEO TO 512-D REPRESENTATION)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN

    v_nodes = [
        ("Input Video", "16 Sampled Frames"),
        ("ResNet18 CNN", "ImageNet Backbone"),
        ("Frame Features", "16 × 512-D Vectors"),
        ("Mean Pooling", "Temporal Average"),
        ("StandardScaler", "512-D Visual Vector")
    ]
    node_w4 = Inches(1.9)
    node_h4 = Inches(0.85)
    node_gap4 = Inches(0.5)

    for i, (ntitle, nsub) in enumerate(v_nodes):
        n_left = Inches(1.0) + i * (node_w4 + node_gap4)
        add_flow_node(slide4, n_left, Inches(1.9), node_w4, node_h4, ntitle, nsub, bg_color=COLOR_NODE_BG, border_color=COLOR_CYAN)
        if i < len(v_nodes) - 1:
            add_arrow_label(slide4, n_left + node_w4, Inches(1.9), node_gap4, node_h4, "──►")

    # Section 2: Language Branch Box
    add_card(slide4, Inches(0.8), Inches(3.35), Inches(11.7), Inches(1.75), bg_color=COLOR_CARD)
    tb_lh = slide4.shapes.add_textbox(Inches(1.0), Inches(3.4), Inches(11.3), Inches(0.3))
    tf_lh = tb_lh.text_frame
    p = tf_lh.paragraphs[0]
    p.text = "NATURAL LANGUAGE BRANCH (ACTION PROMPTS TO 384-D SEMANTICS)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_SKY

    l_nodes = [
        ("45 Descriptions", "5 Prompts per Class"),
        ("Sentence-BERT", "all-MiniLM-L6-v2"),
        ("Text Embeddings", "45 × 384-D Vectors"),
        ("Class Averaging", "Per-Class Mean"),
        ("L2 Normalization", "9 × 384-D Space")
    ]

    for i, (ntitle, nsub) in enumerate(l_nodes):
        n_left = Inches(1.0) + i * (node_w4 + node_gap4)
        add_flow_node(slide4, n_left, Inches(3.8), node_w4, node_h4, ntitle, nsub, bg_color=COLOR_NODE_BG, border_color=COLOR_SKY)
        if i < len(l_nodes) - 1:
            add_arrow_label(slide4, n_left + node_w4, Inches(3.8), node_gap4, node_h4, "──►")

    # Section 3: Fusion & XAI Module Box
    add_card(slide4, Inches(0.8), Inches(5.25), Inches(11.7), Inches(1.8), bg_color=COLOR_CARD, border_color=COLOR_GREEN)
    tb_fh = slide4.shapes.add_textbox(Inches(1.0), Inches(5.3), Inches(11.3), Inches(0.3))
    tf_fh = tb_fh.text_frame
    p = tf_fh.paragraphs[0]
    p.text = "CROSS-MODAL PROJECTION & EXPLAINABLE INFERENCE (ZERO LABEL LEAKAGE)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_GREEN

    f_nodes = [
        ("Visual Vector", "512-D Input"),
        ("Projection Layer", "Linear 512 → 384-D"),
        ("Predicted Vector", "384-D in Text Space"),
        ("Cosine Matching", "vs 9 Class Anchors"),
        ("XAI Engine", "Top-K & Explanation")
    ]

    for i, (ntitle, nsub) in enumerate(f_nodes):
        n_left = Inches(1.0) + i * (node_w4 + node_gap4)
        add_flow_node(slide4, n_left, Inches(5.7), node_w4, node_h4, ntitle, nsub, bg_color=COLOR_NODE_BG, border_color=COLOR_GREEN)
        if i < len(f_nodes) - 1:
            add_arrow_label(slide4, n_left + node_w4, Inches(5.7), node_gap4, node_h4, "──►")

    set_speaker_notes(slide4,
        "SPEAKER NOTES (Slide 4 — System Architecture):\n"
        "• DUAL-BRANCH PARADIGM: Our system is composed of two distinct branches that interface at the inference stage.\n"
        "• VISUAL BRANCH: An input video's 16 frames are passed through ResNet18 to obtain 512-D frame embeddings, which are temporally averaged into a single 512-D video vector and standardized.\n"
        "• LANGUAGE BRANCH: In parallel, 45 prompt descriptions (5 per class) are encoded using Sentence-BERT (all-MiniLM-L6-v2) into 384-D vectors, averaged per class, and L2-normalized into canonical class semantic vectors.\n"
        "• FUSION & INFERENCE: During inference, only the video is supplied. The Semantic Projection Network maps the 512-D visual feature into the 384-D semantic space, where it is matched against all 9 class anchors via cosine similarity.\n"
        "• TRANSITION: Before looking at fusion, let us evaluate the visual baseline and semantic space separately."
    )

    # =========================================================================
    # SLIDE 5: Visual Baseline & Semantic Representation
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "Visual Baseline & Semantic Representation", "Modality Analysis")

    col_w5 = Inches(5.6)
    add_card(slide5, Inches(0.8), Inches(1.45), col_w5, Inches(5.4), bg_color=COLOR_CARD)
    tb_vb = slide5.shapes.add_textbox(Inches(1.0), Inches(1.6), col_w5 - Inches(0.4), Inches(5.1))
    tf_vb = tb_vb.text_frame
    tf_vb.word_wrap = True

    p = tf_vb.paragraphs[0]
    p.text = "VIDEO-ONLY CNN BASELINE"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN

    p = tf_vb.add_paragraph()
    p.text = "Architecture: ResNet18 (512-D) + Mean Pooling + StandardScaler + Logistic Regression (L-BFGS)"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_TEXT_MUTED
    p.space_before = Pt(4)

    p = tf_vb.add_paragraph()
    p.text = "Test Accuracy: 93.40%  |  Macro F1: 92.77%"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_GREEN
    p.space_before = Pt(8)

    p = tf_vb.add_paragraph()
    p.text = "• Perfect 1.0000 F1 Classes: Bowling, Drumming, PlayingGuitar, Typing (High visual distinctiveness).\n• Strong Classes: Biking (0.9756), Archery (0.9167), RopeClimbing (0.8889).\n• Difficult Classes: Basketball (0.7907), JavelinThrow (0.7778).\n• Common Misclassifications: JavelinThrow → Basketball (4 clips), Archery → JavelinThrow (3 clips) due to shared standing poses & arm elevation."
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_TEXT_LIGHT
    p.space_before = Pt(8)

    # Right Column: NLP Semantic Representation + Heatmap
    right_x5 = Inches(6.8)
    add_card(slide5, right_x5, Inches(1.45), col_w5, Inches(5.4), bg_color=COLOR_CARD)
    tb_nlp = slide5.shapes.add_textbox(right_x5 + Inches(0.2), Inches(1.6), col_w5 - Inches(0.4), Inches(1.3))
    tf_nlp = tb_nlp.text_frame
    tf_nlp.word_wrap = True

    p = tf_nlp.paragraphs[0]
    p.text = "SENTENCE-BERT SEMANTIC ACTION SPACE"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_SKY

    p = tf_nlp.add_paragraph()
    p.text = "Model: all-MiniLM-L6-v2 (384-D) | 45 descriptions (5/class) | Metric: Cosine Similarity\n• Highest pairs: Drumming ↔ Guitar (0.5231), Basketball ↔ Bowling (0.4992)\n• Lowest pairs: Javelin ↔ Typing (0.1454), Bowling ↔ Typing (0.1727)"
    p.font.name = FONT_BODY
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_TEXT_LIGHT
    p.space_before = Pt(4)

    # Embed Actual Heatmap Image
    if HEATMAP_PATH.exists():
        slide5.shapes.add_picture(str(HEATMAP_PATH), right_x5 + Inches(0.5), Inches(2.95), width=Inches(4.6), height=Inches(3.7))

    set_speaker_notes(slide5,
        "SPEAKER NOTES (Slide 5 — Baseline & Semantic Space):\n"
        "• VISUAL BASELINE PERFORMANCE: The video-only baseline achieved a very strong 93.40% test accuracy (184 of 197 correct) and 92.77% Macro F1.\n"
        "• ANALYSIS: Actions with distinct objects—like instruments in Guitar and Drums, lanes in Bowling, or keyboards in Typing—reach a perfect 1.0000 F1 score. Confusions occurred primarily between JavelinThrow and Basketball (4 clips) and Archery and Javelin (3 clips) due to similar body poses and raised arms.\n"
        "• SEMANTIC HEATMAP: On the right, the Sentence-BERT cosine similarity heatmap demonstrates that natural language embeddings capture genuine semantic closeness. Drumming and PlayingGuitar share the highest similarity at 0.5231, while Typing and JavelinThrow are furthest at 0.1454.\n"
        "• KEY DETAIL: Notice that Archery and JavelinThrow are semantically close (0.4892). This exact semantic closeness will re-appear in our multimodal fusion model.\n"
        "• TRANSITION: Now let us discuss the critical scientific challenge we uncovered: Label Leakage in naive fusion."
    )

    # =========================================================================
    # SLIDE 6: Vision-Language Fusion & Label Leakage
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "Vision-Language Fusion & Label Leakage Dilemma", "Scientific Rigor")

    card_w6 = Inches(5.6)
    top_y6 = Inches(1.45)

    # Left: Invalid / Oracle Fusion Card
    add_card(slide6, Inches(0.8), top_y6, card_w6, Inches(4.3), bg_color=COLOR_CARD, border_color=COLOR_RED)
    tb_o = slide6.shapes.add_textbox(Inches(1.0), top_y6 + Inches(0.15), card_w6 - Inches(0.4), Inches(4.0))
    tf_o = tb_o.text_frame
    tf_o.word_wrap = True

    p = tf_o.paragraphs[0]
    p.text = "❌ INVALID ORACLE FUSION (TARGET LEAKAGE)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_RED

    p = tf_o.add_paragraph()
    p.text = "Video Feature (512-D) + Ground-Truth Text Embedding (384-D) ──► Concatenation (896-D) ──► 100.0% Test Accuracy"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEXT_WHITE
    p.space_before = Pt(8)

    p = tf_o.add_paragraph()
    p.text = "• Fatal Methodological Flaw: The model is provided the Sentence-BERT embedding of the TRUE ground-truth action label at test time.\n• Artificially Perfect Score: The classifier simply reads the label embedded in the text representation.\n• Non-Deployable: In production on unseen video, the ground-truth text label is unknown and unavailable!"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_TEXT_LIGHT
    p.space_before = Pt(8)

    # Right: Leakage-Safe Semantic Projection Card
    right_x6 = Inches(6.8)
    add_card(slide6, right_x6, top_y6, card_w6, Inches(4.3), bg_color=COLOR_CARD, border_color=COLOR_GREEN)
    tb_s = slide6.shapes.add_textbox(right_x6 + Inches(0.2), top_y6 + Inches(0.15), card_w6 - Inches(0.4), Inches(4.0))
    tf_s = tb_s.text_frame
    tf_s.word_wrap = True

    p = tf_s.paragraphs[0]
    p.text = "✅ LEAKAGE-SAFE SEMANTIC PROJECTION (DEPLOYABLE)"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_GREEN

    p = tf_s.add_paragraph()
    p.text = "Video Feature (512-D) ──► Learned Linear Projection ──► Predicted Semantic Vector (384-D) ──► 87.82% Test Accuracy"
    p.font.name = FONT_HEADING
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_TEXT_WHITE
    p.space_before = Pt(8)

    p = tf_s.add_paragraph()
    p.text = "• Rigorous Zero-Leakage Design: The model receives ONLY video frames during inference. No ground-truth label or text embedding is provided!\n• Alignment Mechanism: Uses cosine distance loss during training to align visual space into semantic space.\n• Truly Deployable: Validated on 197 held-out test videos, achieving genuine, honest multimodal prediction."
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_TEXT_LIGHT
    p.space_before = Pt(8)

    # Bottom Banner Quote
    banner = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.95), Inches(11.7), Inches(0.95))
    banner.fill.solid()
    banner.fill.fore_color.rgb = RGBColor(20, 30, 48)
    banner.line.color.rgb = COLOR_CYAN
    banner.line.width = Pt(1.5)
    tf_b = banner.text_frame
    tf_b.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_b = tf_b.paragraphs[0]
    p_b.text = "SCIENTIFIC INTEGRITY PRINCIPLE: \"High accuracy is meaningless if the evaluation pipeline leaks ground-truth information. Rigorous ML demands testing only with information available at deployment.\""
    p_b.alignment = PP_ALIGN.CENTER
    p_b.font.name = FONT_HEADING
    p_b.font.size = Pt(12)
    p_b.font.bold = True
    p_b.font.color.rgb = COLOR_SKY

    set_speaker_notes(slide6,
        "SPEAKER NOTES (Slide 6 — Vision-Language Fusion & Label Leakage):\n"
        "• CRITICAL SCIENTIFIC INSIGHT: In Week 4 of our project, we explored multimodal fusion. A naive early-fusion approach concatenates visual features with text embeddings. When tested, this scored an artificial 100.0% accuracy.\n"
        "• WHY 100% IS FLAWED: This is an Oracle result caused by Target Label Leakage. By feeding the true action description embedding at test time, the model doesn't recognize video—it just reads the answer from the text.\n"
        "• THE PROPER SOLUTION: We formulated the Leakage-Safe Semantic Projection architecture. The network maps visual features into the 384-D semantic space without any access to ground-truth labels during inference. It predicts actions by cosine similarity against fixed class anchors.\n"
        "• TAKEAWAY: Reporting 87.82% honestly is far more valuable and scientifically sound than showcasing an artificial 100% oracle score. Rigorous research requires eliminating leakage completely.\n"
        "• TRANSITION: Let us examine the full ablation study comparing all these models."
    )

    # =========================================================================
    # SLIDE 7: Experimental Results & Ablation
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "Experimental Results & Ablation Analysis", "Quantitative Evaluation")

    # Table of Results
    rows = 5
    cols = 6
    left_t = Inches(0.8)
    top_t = Inches(1.45)
    width_t = Inches(11.7)
    height_t = Inches(2.7)

    table_shape = slide7.shapes.add_table(rows, cols, left_t, top_t, width_t, height_t)
    table = table_shape.table

    table.columns[0].width = Inches(3.2)  # Model
    table.columns[1].width = Inches(1.4)  # Feature Dim
    table.columns[2].width = Inches(1.8)  # Leakage Safe?
    table.columns[3].width = Inches(1.8)  # Deployable?
    table.columns[4].width = Inches(1.7)  # Test Accuracy
    table.columns[5].width = Inches(1.8)  # Macro F1

    table_data = [
        ["Model Configuration", "Feature Dim", "Label Leakage?", "Deployable?", "Test Accuracy", "Macro F1"],
        ["Video-Only Baseline (ResNet18 + LogReg)", "512-D", "No Leakage", "✅ Yes", "93.40%", "92.77%"],
        ["Text-Only Oracle (Sentence-BERT)", "384-D", "⚠️ YES (Oracle)", "❌ No", "100.00%*", "100.00%*"],
        ["Oracle Concatenation (Visual + True Text)", "896-D", "⚠️ YES (Oracle)", "❌ No", "100.00%*", "100.00%*"],
        ["Leakage-Safe Semantic Projection", "384-D", "No Leakage", "✅ Yes", "87.82%", "86.39%"]
    ]

    for r_idx, row in enumerate(table_data):
        for c_idx, cell_value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = cell_value
            p = cell.text_frame.paragraphs[0]
            p.font.name = FONT_HEADING
            p.font.size = Pt(11)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

            if r_idx == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(15, 30, 55)
                p.font.bold = True
                p.font.color.rgb = COLOR_CYAN
                p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            else:
                cell.fill.solid()
                if r_idx in (2, 3):
                    cell.fill.fore_color.rgb = RGBColor(38, 20, 25)
                else:
                    cell.fill.fore_color.rgb = COLOR_CARD if r_idx % 2 == 1 else COLOR_CARD_ALT

                p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
                if c_idx == 0:
                    p.font.bold = True
                    p.font.color.rgb = COLOR_TEXT_WHITE
                elif c_idx in (4, 5):
                    p.font.bold = True
                    p.font.color.rgb = COLOR_GREEN if r_idx in (1, 4) else COLOR_RED
                else:
                    p.font.color.rgb = COLOR_TEXT_LIGHT

    note_box = slide7.shapes.add_textbox(Inches(0.8), Inches(4.2), Inches(11.7), Inches(0.4))
    tf_n = note_box.text_frame
    p = tf_n.paragraphs[0]
    p.text = "*Note: Top-3 Accuracy, Top-5 Accuracy & Inference Latency: [RESULT NOT AVAILABLE] (strictly omitted per verified repository evidence)."
    p.font.name = FONT_BODY
    p.font.size = Pt(9.5)
    p.font.color.rgb = COLOR_TEXT_MUTED

    # 3 Takeaway Cards Across Bottom
    card_w7 = Inches(3.75)
    card_h7 = Inches(2.2)
    card_gap7 = Inches(0.22)
    top_y7 = Inches(4.65)

    takeaways = [
        ("Honest Trade-Off (-5.58%)", "Why Baseline is Higher", "The visual baseline operates directly on 512-D visual features, whereas linear projection compresses features into a general 384-D language space without fine-tuning ResNet18."),
        ("Preserved Zero-Shot Capability", "Semantic Flexibility", "Despite the slight drop, the projection model classifies purely by semantic proximity, allowing open-ended prompt matching and zero-shot reasoning without changing architecture."),
        ("Semantic Error Alignment", "Reasoning vs Random Errors", "Projection errors occur between semantically close classes (e.g. Archery ↔ JavelinThrow), proving that errors follow conceptual logic rather than arbitrary classification noise.")
    ]

    for i, (t_title, t_sub, t_desc) in enumerate(takeaways):
        left = Inches(0.8) + i * (card_w7 + card_gap7)
        add_card(slide7, left, top_y7, card_w7, card_h7, bg_color=COLOR_CARD)
        tb = slide7.shapes.add_textbox(left + Inches(0.15), top_y7 + Inches(0.15), card_w7 - Inches(0.3), card_h7 - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = t_title.upper()
        p.font.name = FONT_HEADING
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_SKY

        p2 = tf.add_paragraph()
        p2.text = t_sub
        p2.font.name = FONT_HEADING
        p2.font.size = Pt(12)
        p2.font.bold = True
        p2.font.color.rgb = COLOR_TEXT_WHITE
        p2.space_before = Pt(2)

        p3 = tf.add_paragraph()
        p3.text = t_desc
        p3.font.name = FONT_BODY
        p3.font.size = Pt(10)
        p3.font.color.rgb = COLOR_TEXT_MUTED
        p3.space_before = Pt(4)

    set_speaker_notes(slide7,
        "SPEAKER NOTES (Slide 7 — Experimental Results & Ablation):\n"
        "• WALKTHROUGH: This table summarizes all four experimental setups evaluated on the 197 held-out test videos.\n"
        "• ORACLE MODELS: Rows 2 and 3 highlight the Text-Only Oracle and Oracle Concatenation reaching 100% accuracy. We clearly label these as non-deployable due to label leakage.\n"
        "• DEPLOYABLE COMPARISON: Looking at genuine deployable models, the Video-Only baseline achieves 93.40% accuracy and 92.77% Macro F1, while the Leakage-Safe Semantic Projection achieves 87.82% accuracy and 86.39% Macro F1.\n"
        "• SCIENTIFIC HONESTY: We do not hide the 5.58% accuracy difference. Projecting a 512-D visual feature into a generic 384-D text space naturally compresses visual variance. However, it gains semantic interpretability, prompt guidance, and zero-shot matching.\n"
        "• TRANSITION: Let us examine how this semantic projection enables real-time Explainable AI on test videos."
    )

    # =========================================================================
    # SLIDE 8: Explainable Prediction (XAI Case Study)
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "Explainable AI & Qualitative Prediction Case Study", "Decision Transparency")

    card_w8 = Inches(5.6)
    add_card(slide8, Inches(0.8), Inches(1.45), card_w8, Inches(5.4), bg_color=COLOR_CARD)
    tb_ve = slide8.shapes.add_textbox(Inches(1.0), Inches(1.6), card_w8 - Inches(0.4), Inches(0.8))
    tf_ve = tb_ve.text_frame
    tf_ve.word_wrap = True
    p = tf_ve.paragraphs[0]
    p.text = "VISUAL EVIDENCE: SAMPLED FRAMES CONTACT SHEET"
    p.font.name = FONT_HEADING
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN

    p = tf_ve.add_paragraph()
    p.text = "Held-Out Test Video: v_PlayingGuitar_g10_c01.avi (16 sampled frames)"
    p.font.name = FONT_BODY
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_TEXT_MUTED
    p.space_before = Pt(2)

    # Embed Actual Contact Sheet Image
    if CONTACT_SHEET_PATH.exists():
        slide8.shapes.add_picture(str(CONTACT_SHEET_PATH), Inches(1.0), Inches(2.35), width=Inches(5.2), height=Inches(4.25))

    # Right: Structured XAI Prediction Report
    right_x8 = Inches(6.8)
    add_card(slide8, right_x8, Inches(1.45), card_w8, Inches(5.4), bg_color=COLOR_CARD)
    tb_rep = slide8.shapes.add_textbox(right_x8 + Inches(0.2), Inches(1.6), card_w8 - Inches(0.4), Inches(5.1))
    tf_rep = tb_rep.text_frame
    tf_rep.word_wrap = True

    p = tf_rep.paragraphs[0]
    p.text = "STRUCTURED XAI PREDICTION REPORT"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_SKY

    p = tf_rep.add_paragraph()
    p.text = "Predicted Action: PlayingGuitar   |   Confidence: Moderate"
    p.font.name = FONT_HEADING
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_GREEN
    p.space_before = Pt(6)

    p = tf_rep.add_paragraph()
    p.text = "Top-1 Cosine Similarity: 0.5230   |   Score Separation: +0.4719 (Clear)"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_TEXT_WHITE
    p.space_before = Pt(2)

    p = tf_rep.add_paragraph()
    p.text = "TOP-3 RANKED CANDIDATE ACTIONS:"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN
    p.space_before = Pt(8)

    p = tf_rep.add_paragraph()
    p.text = "1. PlayingGuitar:  0.5230  (Top Match — Clear Lead)\n2. Drumming:       0.0511  (Runner-up — Rhythmic Semantics)\n3. RopeClimbing:  -0.0851  (Distant Candidate)"
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.color.rgb = COLOR_TEXT_LIGHT
    p.space_before = Pt(4)

    p = tf_rep.add_paragraph()
    p.text = "NATURAL LANGUAGE EXPLANATION:"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_SKY
    p.space_before = Pt(8)

    p = tf_rep.add_paragraph()
    p.text = "\"The projected semantic representation of the video was most similar to the PlayingGuitar class embedding. Its similarity score of 0.5230 was clearly higher than the next-best class, Drumming, which scored 0.0511.\""
    p.font.name = FONT_BODY
    p.font.size = Pt(10.5)
    p.font.italic = True
    p.font.color.rgb = COLOR_TEXT_LIGHT
    p.space_before = Pt(4)

    p = tf_rep.add_paragraph()
    p.text = "RELIABILITY NOTE:"
    p.font.name = FONT_HEADING
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_AMBER
    p.space_before = Pt(8)

    p = tf_rep.add_paragraph()
    p.text = "\"These values are cosine similarity scores, not calibrated probabilities. The top result has a reasonably distinct semantic advantage (+0.4719) over the next candidate.\""
    p.font.name = FONT_BODY
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_TEXT_MUTED
    p.space_before = Pt(4)

    set_speaker_notes(slide8,
        "SPEAKER NOTES (Slide 8 — Explainable AI Case Study):\n"
        "• QUALITATIVE EVIDENCE: Slide 8 demonstrates our Explainability Engine on a real test video: `v_PlayingGuitar_g10_c01.avi`.\n"
        "• VISUAL AUDITABILITY: On the left, 16 sampled frames are automatically compiled into a contact sheet, giving users and auditors immediate visual verification of what the model saw.\n"
        "• XAI REPORT: On the right, the system generates a structured explanation report. PlayingGuitar leads with 0.5230 similarity. The runner-up is Drumming at 0.0511—which aligns with our earlier Sentence-BERT finding that Drumming and Guitar are semantic neighbours.\n"
        "• SCORE SEPARATION: The score gap is +0.4719, triggering the 'Clear Separation' category and dynamically generating the English explanation text.\n"
        "• RELIABILITY NOTE: It explicitly informs the user that similarity scores reflect relative directional alignment, avoiding misinterpretation as Bayesian probabilities.\n"
        "• TRANSITION: All of these capabilities are integrated into our interactive Streamlit application."
    )

    # =========================================================================
    # SLIDE 9: Streamlit Application & Deployment
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "Interactive Deployment — Streamlit Web Application", "Application Engineering")

    col_w9 = Inches(5.6)

    # Left Column: Streamlit UI Mockup / Flow Showcase
    add_card(slide9, Inches(0.8), Inches(1.45), col_w9, Inches(5.4), bg_color=COLOR_CARD)
    tb_ui = slide9.shapes.add_textbox(Inches(1.0), Inches(1.6), col_w9 - Inches(0.4), Inches(5.1))
    tf_ui = tb_ui.text_frame
    tf_ui.word_wrap = True

    p = tf_ui.paragraphs[0]
    p.text = "STREAMLIT UI & INTERACTIVE PIPELINE"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_CYAN

    ui_steps = [
        ("1. Video Upload", "Supports .mp4, .avi, .mov, and .mkv video formats."),
        ("2. Interactive Configuration", "Sidebar sliders for Top-K candidates (1–9) and frame sampling (4–32)."),
        ("3. Automated Execution", "OpenCV sampling → ResNet18 extraction → Semantic projection."),
        ("4. Explainable Output Card", "Predicted action, confidence level badge, and score separation."),
        ("5. Semantic Proximity Ranking", "Top-K bar charts & natural-language justification."),
        ("6. Downloadable Artifacts", "Machine-readable JSON reports & visual frame contact sheets.")
    ]

    for title, desc in ui_steps:
        p = tf_ui.add_paragraph()
        p.text = f"• {title}:"
        p.font.name = FONT_HEADING
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(5)

        p_d = tf_ui.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.name = FONT_BODY
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = COLOR_TEXT_LIGHT
        p_d.space_before = Pt(1)

    # Right Column: Technology Stack Badges & Architecture
    right_x9 = Inches(6.8)
    add_card(slide9, right_x9, Inches(1.45), col_w9, Inches(5.4), bg_color=COLOR_CARD)
    tb_tech = slide9.shapes.add_textbox(right_x9 + Inches(0.2), Inches(1.6), col_w9 - Inches(0.4), Inches(5.1))
    tf_tech = tb_tech.text_frame
    tf_tech.word_wrap = True

    p = tf_tech.paragraphs[0]
    p.text = "MODULAR TECHNOLOGY STACK"
    p.font.name = FONT_HEADING
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_SKY

    tech_stack = [
        ("Python 3.12 & PyTorch 2.13", "Deep learning backbone, tensor processing, and projection model execution."),
        ("Torchvision & ResNet18", "Pretrained CNN feature extraction (512-D ImageNet visual representation)."),
        ("Sentence-Transformers", "Sentence-BERT (all-MiniLM-L6-v2) for 384-D semantic prompt encoding."),
        ("OpenCV (cv2)", "Fast video decoding, uniform temporal frame sampling, and contact sheet generation."),
        ("Scikit-Learn", "StandardScaler feature normalization and baseline Logistic Regression models."),
        ("Streamlit Web Framework", "Interactive, reactive web interface and visualization dashboard."),
        ("Git & GitHub", "Version control, experiment tracking, and open-source reproducibility.")
    ]

    for title, desc in tech_stack:
        p = tf_tech.add_paragraph()
        p.text = f"• {title}:"
        p.font.name = FONT_HEADING
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(4)

        p_d = tf_tech.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.name = FONT_BODY
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = COLOR_TEXT_MUTED
        p_d.space_before = Pt(1)

    set_speaker_notes(slide9,
        "SPEAKER NOTES (Slide 9 — Streamlit Application):\n"
        "• APPLICATION HIGHLIGHT: In Week 6, we brought the entire 6-week research effort into an interactive Streamlit web application (`app/streamlit_app.py`).\n"
        "• USER EXPERIENCE: Instead of running command-line scripts, a user simply drags and drops any standard video file (.mp4, .avi, .mov). They can customize the Top-K candidate count and frame sampling rate via sidebar sliders.\n"
        "• BACKEND EXECUTION: The application executes the complete end-to-end pipeline: OpenCV frame sampling, ResNet18 feature extraction, semantic projection, cosine matching against the 9 class embeddings, and the XAI report generation.\n"
        "• TECH STACK: The implementation is completely modular and reproducible, leveraging PyTorch, Sentence-Transformers, OpenCV, Scikit-learn, and Streamlit.\n"
        "• TRANSITION: To conclude our presentation, let us review our contributions, limitations, and future research directions."
    )

    # =========================================================================
    # SLIDE 10: Conclusion, Limitations & Future Work
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "Conclusion, Limitations & Future Roadmap", "Project Summary")

    col_w10 = Inches(3.75)
    gap10 = Inches(0.22)
    top_y10 = Inches(1.45)
    height10 = Inches(4.5)

    sections = [
        ("PROJECT CONTRIBUTIONS", COLOR_GREEN, [
            "• End-to-End Multimodal System: Successfully bridged CNN video features with Sentence-BERT language representations.",
            "• Methodological Rigor: Identified and eliminated label leakage, demonstrating honest 87.82% deployable test accuracy.",
            "• Explainable Predictions: Created structured XAI reports with similarity ranking, score separation, and confidence notes.",
            "• Production Web App: Deployed interactive Streamlit interface for demonstration and testing."
        ]),
        ("PROJECT LIMITATIONS", COLOR_AMBER, [
            "• Dataset Subset: Evaluated on 9 curated classes of UCF101 rather than the full 101-class distribution.",
            "• Mean Pooling Limitation: Frame averaging discards temporal ordering, motion speed, and phase transitions.",
            "• Linear Projection: A single linear layer limits complex nonlinear cross-modal alignment.",
            "• Uncalibrated Scores: Cosine similarity provides relative rankings rather than calibrated Bayesian probabilities."
        ]),
        ("FUTURE EXTENSIONS", COLOR_SKY, [
            "• Video Transformers: Integrate TimeSformer or Video Swin Transformer for rich temporal dynamics modeling.",
            "• Foundation VLM Models: Explore zero-shot action recognition using Video-CLIP or X-CLIP architectures.",
            "• Dataset Scaling: Expand evaluation to full UCF101, HMDB51, and Kinetics-400 benchmarks.",
            "• Real-Time Edge Deployment: Optimize pipeline for live webcam streams and cloud containerization."
        ])
    ]

    for i, (sec_title, sec_color, bullets) in enumerate(sections):
        left = Inches(0.8) + i * (col_w10 + gap10)
        add_card(slide10, left, top_y10, col_w10, height10, bg_color=COLOR_CARD)
        tb = slide10.shapes.add_textbox(left + Inches(0.15), top_y10 + Inches(0.15), col_w10 - Inches(0.3), height10 - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = sec_title
        p.font.name = FONT_HEADING
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = sec_color

        for bullet in bullets:
            p = tf.add_paragraph()
            p.text = bullet
            p.font.name = FONT_BODY
            p.font.size = Pt(10.5)
            p.font.color.rgb = COLOR_TEXT_LIGHT
            p.space_before = Pt(6)

    # Bottom Closing Card
    close_banner = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.1), Inches(11.7), Inches(0.85))
    close_banner.fill.solid()
    close_banner.fill.fore_color.rgb = RGBColor(20, 30, 48)
    close_banner.line.color.rgb = COLOR_CYAN
    close_banner.line.width = Pt(1.5)
    tf_cb = close_banner.text_frame
    tf_cb.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_cb = tf_cb.paragraphs[0]
    p_cb.text = "THANK YOU!   —   QUESTIONS & DISCUSSION"
    p_cb.alignment = PP_ALIGN.CENTER
    p_cb.font.name = FONT_HEADING
    p_cb.font.size = Pt(16)
    p_cb.font.bold = True
    p_cb.font.color.rgb = COLOR_CYAN

    set_speaker_notes(slide10,
        "SPEAKER NOTES (Slide 10 — Conclusion & Future Work):\n"
        "• CONCLUSION: In summary, this project successfully developed an end-to-end, explainable vision-language action recognition system on UCF101.\n"
        "• KEY RESEARCH CONTRIBUTION: Our most significant contribution is demonstrating that semantic prompt alignment is achievable without compromising scientific integrity. By exposing and eliminating label leakage, we delivered a valid 87.82% deployable model accompanied by human-readable explanations.\n"
        "• FUTURE DIRECTIONS: The limitations of temporal mean pooling and linear projection pave the way for exciting future extensions with Video Transformers (TimeSformer) and Vision-Language Foundation models (CLIP).\n"
        "• CLOSING: Thank you for your time and attention. I now welcome any questions, comments, or feedback from the committee."
    )

    # Save presentation
    prs.save(str(OUTPUT_PPTX))
    print(f"Successfully generated: {OUTPUT_PPTX}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    build_presentation()
