# Makefile for medical paper writing workflow
# Usage: make <target>

# Variables
PAPER_DIR = paper
SCRIPTS_DIR = scripts
OUTPUT_DIR = output
TEMPLATES_DIR = templates
REFERENCE_STYLE_DIR = reference-style

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "Build:"
	@echo "  paper        - Build main paper.docx"
	@echo "  cover        - Build cover-letter.docx"
	@echo "  supplements  - Build supplements.docx"
	@echo "  all          - Build all documents (paper, cover, supplements)"
	@echo ""
	@echo "Copy:"
	@echo "  cp-all        - Copy main paper content (excludes supplements & cover)"
	@echo "  cp-main       - Copy abstract, introduction, methods, results, discussion"
	@echo "  cp-intro      - Copy introduction section"
	@echo "  cp-methods    - Copy methods section"
	@echo "  cp-results    - Copy results section"
	@echo "  cp-discussion - Copy discussion section"
	@echo "  cp-tables     - Copy tables section"
	@echo "  cp-figures    - Copy figures section"
	@echo ""
	@echo "LLM Prompts (prepend to current clipboard):"
	@echo "  pr-correction - Prepend correction prompt to current clipboard"
	@echo "  pr-coherence  - Prepend coherence prompt to current clipboard"
	@echo "  pr-improve    - Prepend improve English prompt to current clipboard"
	@echo "  pr-reference  - Prepend reference finding prompt to current clipboard"
	@echo "  pr-cover      - Prepend cover letter generation prompt to current clipboard"
	@echo "  pr-generate   - Prepend general writing generation prompt to current clipboard"
	@echo "  pr-table     - Prepend abbreviations generation prompt to current clipboard"
	@echo "  pr-figures    - Prepend figure generation prompt to current clipboard"
	@echo ""
	@echo "Section Writing (generation + structure):"
	@echo "  pr-intro      - Copy introduction writing prompt to clipboard"
	@echo "  pr-methods    - Copy methods writing prompt to clipboard"
	@echo "  pr-results    - Copy results writing prompt to clipboard"
	@echo "  pr-discussion - Copy discussion writing prompt to clipboard"
	@echo ""
	@echo "Utility:"
	@echo "  clean        - Remove generated documents"
	@echo "  counts       - Update word counts and content statistics"

# Build main paper
.PHONY: paper
paper: ## Build main paper.docx
	@echo "Building main paper..."
	python3 $(SCRIPTS_DIR)/build_paper.py
	@echo "✅ Main paper built successfully!"

# Build cover letter
.PHONY: cover
cover: ## Build cover-letter.docx
	@echo "Building cover letter..."
	@python3 $(SCRIPTS_DIR)/build_cover.py
	@echo "✅ Cover letter built successfully!"

# Build supplements
.PHONY: supplements
supplements: ## Build supplements.docx
	@echo "Building supplements..."
	@python3 $(SCRIPTS_DIR)/build_supplements.py
	@echo "✅ Supplements built successfully!"

# Build all documents
.PHONY: all
all: paper cover supplements ## Build all documents (paper, cover, supplements)
	@echo "🎉 All documents built successfully!"

# Copy commands for different sections
.PHONY: cp-all cp-main cp-intro cp-methods cp-results cp-discussion cp-tables cp-figures

cp-all: ## Copy main paper content to clipboard (excludes supplements & cover)
	@echo "Copying main paper content..."
	@cat $(PAPER_DIR)/0_title-page.md $(PAPER_DIR)/1_abstract.md $(PAPER_DIR)/2_introduction.md $(PAPER_DIR)/3_methods.md $(PAPER_DIR)/4_results.md $(PAPER_DIR)/5_discussion.md $(PAPER_DIR)/6_figures.md $(PAPER_DIR)/7_table.md $(PAPER_DIR)/9_end.md | pbcopy
	@echo "✅ Main paper content copied to clipboard!"

cp-main: ## Copy abstract, introduction, methods, results, discussion to clipboard
	@echo "Copying abstract through discussion sections..."
	@cat $(PAPER_DIR)/1_abstract.md $(PAPER_DIR)/2_introduction.md $(PAPER_DIR)/3_methods.md $(PAPER_DIR)/4_results.md $(PAPER_DIR)/5_discussion.md | pbcopy
	@echo "✅ Abstract through discussion sections copied to clipboard!"

cp-intro: ## Copy introduction section to clipboard
	@echo "Copying introduction..."
	@cat $(PAPER_DIR)/2_introduction.md | pbcopy
	@echo "✅ Introduction copied to clipboard!"

cp-methods: ## Copy methods section to clipboard
	@echo "Copying methods..."
	@cat $(PAPER_DIR)/3_methods.md | pbcopy
	@echo "✅ Methods copied to clipboard!"

cp-results: ## Copy results section to clipboard
	@echo "Copying results..."
	@cat $(PAPER_DIR)/4_results.md | pbcopy
	@echo "✅ Results copied to clipboard!"

cp-discussion: ## Copy discussion section to clipboard
	@echo "Copying discussion..."
	@cat $(PAPER_DIR)/5_discussion.md | pbcopy
	@echo "✅ Discussion copied to clipboard!"

cp-tables: ## Copy tables section to clipboard
	@echo "Copying tables..."
	@cat $(PAPER_DIR)/7_table.md | pbcopy
	@echo "✅ Tables copied to clipboard!"

cp-figures: ## Copy figures section to clipboard
	@echo "Copying figures..."
	@cat $(PAPER_DIR)/6_figures.md | pbcopy
	@echo "✅ Figures copied to clipboard!"

# Clean generated files
.PHONY: clean
clean: ## Remove generated documents
	@echo "Cleaning generated documents..."
	@rm -f $(OUTPUT_DIR)/*.docx
	@echo "✅ Generated documents removed!"

# Show current configuration
.PHONY: config
config: ## Show current configuration
	@echo "Current configuration:"
	@python3 -c "import json; config = json.load(open('config.json')); print(f'Citation style: {config[\"citation_style\"]}'); print(f'Template: {config[\"build_settings\"][\"template\"]}'); print(f'Number sections: {config[\"build_settings\"][\"number_sections\"]}')"

# Update word counts and content statistics
.PHONY: counts
counts: ## Update word counts and content statistics in title page
	@echo "Updating word counts and content statistics..."
	@python3 $(SCRIPTS_DIR)/word_counter.py

# LLM Prompt Commands - Prepend to current clipboard
.PHONY: pr-correction pr-coherence pr-improve pr-reference pr-cover pr-generate pr-table pr-figures pr-intro pr-methods pr-results pr-discussion

pr-correction: ## Prepend correction prompt to current clipboard
	@echo "Prepending correction prompt to current clipboard..."
	@current_content=$$(pbpaste 2>/dev/null || echo ""); \
	(cat prompt/correction.md && echo "" && echo "$$current_content") | pbcopy
	@echo "✅ Correction prompt prepended to clipboard!"

pr-coherence: ## Prepend coherence prompt to current clipboard
	@echo "Prepending coherence prompt to current clipboard..."
	@current_content=$$(pbpaste 2>/dev/null || echo ""); \
	(cat prompt/coherence.md && echo "" && echo "$$current_content") | pbcopy
	@echo "✅ Coherence prompt prepended to clipboard!"

pr-improve: ## Prepend improve English prompt to current clipboard
	@echo "Prepending improve English prompt to current clipboard..."
	@current_content=$$(pbpaste 2>/dev/null || echo ""); \
	(cat prompt/improve-english.md && echo "" && echo "$$current_content") | pbcopy
	@echo "✅ Improve English prompt prepended to clipboard!"

pr-reference: ## Prepend reference finding prompt to current clipboard
	@echo "Prepending reference finding prompt to current clipboard..."
	@current_content=$$(pbpaste 2>/dev/null || echo ""); \
	(cat prompt/reference.md && echo "" && echo "$$current_content") | pbcopy
	@echo "✅ Reference finding prompt prepended to clipboard!"

pr-cover: ## Prepend cover letter generation prompt to current clipboard
	@echo "Prepending cover letter generation prompt to current clipboard..."
	@current_content=$$(pbpaste 2>/dev/null || echo ""); \
	(cat prompt/cover-letter.md && echo "" && echo "$$current_content") | pbcopy
	@echo "✅ Cover letter generation prompt prepended to clipboard!"

pr-generate: ## Prepend general writing generation prompt to current clipboard
	@echo "Prepending general writing generation prompt to current clipboard..."
	@current_content=$$(pbpaste 2>/dev/null || echo ""); \
	(cat prompt/generation.md && echo "" && echo "$$current_content") | pbcopy
	@echo "✅ General writing generation prompt prepended to clipboard!"

pr-table: ## Prepend abbreviations generation prompt to current clipboard
	@echo "Prepending abbreviations generation prompt to current clipboard..."
	@current_content=$$(pbpaste 2>/dev/null || echo ""); \
	(cat prompt/abbreviations.md && echo "" && echo "$$current_content") | pbcopy
	@echo "✅ Abbreviations generation prompt prepended to clipboard!"

pr-figures: ## Prepend figure generation prompt to current clipboard
	@echo "Prepending figure generation prompt to current clipboard..."
	@current_content=$$(pbpaste 2>/dev/null || echo ""); \
	(cat prompt/figures.md && echo "" && echo "$$current_content") | pbcopy
	@echo "✅ Figure generation prompt prepended to clipboard!"

# Section Writing Commands - Combine generation + structure prompts
pr-intro: ## Copy introduction writing prompt to clipboard
	@echo "Copying introduction writing prompt to clipboard..."
	@(cat prompt/generation.md && echo "" && cat prompt/introduction.md) | pbcopy
	@echo "✅ Introduction writing prompt copied to clipboard!"

pr-methods: ## Copy methods writing prompt to clipboard
	@echo "Copying methods writing prompt to clipboard..."
	@(cat prompt/generation.md && echo "" && cat prompt/methods.md) | pbcopy
	@echo "✅ Methods writing prompt copied to clipboard!"

pr-results: ## Copy results writing prompt to clipboard
	@echo "Copying results writing prompt to clipboard..."
	@(cat prompt/generation.md && echo "" && cat prompt/results.md) | pbcopy
	@echo "✅ Results writing prompt copied to clipboard!"

pr-discussion: ## Copy discussion writing prompt to clipboard
	@echo "Copying discussion writing prompt to clipboard..."
	@(cat prompt/generation.md && echo "" && cat prompt/discussion.md) | pbcopy
	@echo "✅ Discussion writing prompt copied to clipboard!"
