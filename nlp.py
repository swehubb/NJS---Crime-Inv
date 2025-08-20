import os
import platform
import whisper
from openai import OpenAI
from transformers import pipeline
from empath import Empath
import json
from typing import Dict, List, Tuple, Any
import argparse
from datetime import datetime
import re
import ffmpeg
from dotenv import load_dotenv
import sys

def load_transcript_from_json(file_path: str) -> str:

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        transcript_text = None
        
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and 'text' in data[0]:
                transcript_text = ' '.join([item.get('text', '') for item in data if 'text' in item])
                print(f"Loaded {len(data)} witness statements from array")
            elif isinstance(data[0], dict) and ('text' in data[0] or 'start' in data[0]):
                transcript_text = ' '.join([segment.get('text', '') for segment in data])
        
        elif isinstance(data, dict):
            for field in ['transcript', 'text', 'transcription', 'content']:
                if field in data:
                    if isinstance(data[field], str):
                        transcript_text = data[field]
                        break
                    elif isinstance(data[field], dict) and 'text' in data[field]:
                        transcript_text = data[field]['text']
                        break
            
            if not transcript_text and 'transcript' in data and isinstance(data['transcript'], dict):
                transcript_text = data['transcript'].get('text', '')
            
            if not transcript_text and 'results' in data:
                results = data['results']
                if 'transcript' in results and isinstance(results['transcript'], dict):
                    transcript_text = results['transcript'].get('text', '')
        
        if transcript_text and len(transcript_text.strip()) > 10:
            print(f"Loaded transcript: {len(transcript_text)} characters")
            return transcript_text.strip()
        else:
            print("ERROR: No valid transcript text found in JSON")
            return None
            
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON format: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Error loading JSON: {e}")
        return None

def multi_witness_demo():
    """Interactive demo for analyzing multiple witnesses"""
    print("MULTI-WITNESS ANALYSIS DEMO")
    print("="*50)
    
    print("\nSELECT INPUT METHOD:")
    print("1. Folder with multiple audio files")
    print("2. JSON config file with witness details")
    print("3. Manual input (enter witness details one by one)")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    analyzer = WitnessReportAnalyzer()
    
    if choice == '1':
        folder_path = input("Enter folder path containing audio files: ").strip()
        if not os.path.exists(folder_path):
            print(f"ERROR: Folder not found: {folder_path}")
            return
        
        import glob
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.m4a', '*.flac', '*.ogg', '*.wma', '*.aac']:
            audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if not audio_files:
            print(f"ERROR: No audio files found in {folder_path}")
            return
        
        print(f"Found {len(audio_files)} audio files")
        
        witness_inputs = []
        for i, audio_file in enumerate(audio_files, 1):
            witness_inputs.append({
                'witness_id': f"witness_{i}_{os.path.splitext(os.path.basename(audio_file))[0]}",
                'audio_path': audio_file
            })
        
        use_api = False
        if os.getenv('OPENAI_API_KEY'):
            use_api_input = input("Use OpenAI API for transcription? (y/n, default: n): ").strip().lower()
            use_api = use_api_input in ['y', 'yes']
        
        model_size = input("Whisper model size (tiny/base/small/medium/large, default: base): ").strip() or "base"
        
        print(f"\nAnalyzing {len(witness_inputs)} witnesses...")
        results = analyzer.analyze_multiple_witnesses(
            witness_inputs=witness_inputs,
            use_api=use_api,
            model_size=model_size
        )
        
        print("Analysis complete!")
        print(f"Individual files: {len(results.get('individual_files', []))}")
        print(f"Summary report: {results.get('summary_file')}")
    
    elif choice == '2':
        config_path = input("Enter path to JSON config file: ").strip()
        if not os.path.exists(config_path):
            print(f"ERROR: Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if isinstance(config, list) and len(config) > 0 and 'text' in config[0]:
                print(f"Detected witness statements array format with {len(config)} witnesses")
                witness_inputs = []
                for i, statement in enumerate(config, 1):
                    witness_inputs.append({
                        'witness_id': f"witness_{i}",
                        'transcript': statement.get('text', '')
                    })
            elif isinstance(config, dict):
                witness_inputs = config.get('witnesses', [])
            else:
                witness_inputs = []
            
            if not witness_inputs:
                print("ERROR: No witnesses found in config file")
                return
            
            print(f"Found {len(witness_inputs)} witnesses in config")
            
            results = analyzer.analyze_multiple_witnesses(witness_inputs=witness_inputs)
            print("Analysis complete!")
            print(f"Individual files: {len(results.get('individual_files', []))}")
            print(f"Summary report: {results.get('summary_file')}")
            
        except Exception as e:
            print(f"ERROR: Error reading config file: {e}")
    
    elif choice == '3':
        witnesses = []
        witness_count = 1
        
        while True:
            print(f"\nWITNESS {witness_count}")
            print("-" * 30)
            
            witness_id = input(f"Witness ID (default: witness_{witness_count}): ").strip() or f"witness_{witness_count}"
            
            print("Input method:")
            print("1. Audio file")
            print("2. Text transcript")
            
            input_choice = input("Select (1/2): ").strip()
            
            witness_data = {"witness_id": witness_id}
            
            if input_choice == '1':
                audio_path = input("Audio file path: ").strip()
                if os.path.exists(audio_path):
                    witness_data['audio_path'] = audio_path
                else:
                    print(f"WARNING: File not found: {audio_path}")
                    continue
            elif input_choice == '2':
                print("Enter transcript (press Enter twice when done):")
                lines = []
                while True:
                    line = input()
                    if line == "" and len(lines) > 0 and lines[-1] == "":
                        break
                    lines.append(line)
                transcript = '\n'.join(lines).strip()
                if len(transcript) > 50:
                    witness_data['transcript'] = transcript
                else:
                    print("WARNING: Transcript too short")
                    continue
            else:
                print("ERROR: Invalid choice")
                continue
            
            witnesses.append(witness_data)
            witness_count += 1
            
            more = input(f"\nAdd another witness? (y/n): ").strip().lower()
            if more not in ['y', 'yes']:
                break
        
        if witnesses:
            print(f"\nAnalyzing {len(witnesses)} witnesses...")
            results = analyzer.analyze_multiple_witnesses(witness_inputs=witnesses)
            print("Analysis complete!")
            print(f"Individual files: {len(results.get('individual_files', []))}")
            print(f"Summary report: {results.get('summary_file')}")
        else:
            print("ERROR: No witnesses to analyze")
    
    else:
        print("ERROR: Invalid selection")

def create_sample_witness_config():
    """Create a sample witness configuration file"""
    sample_config = {
        "case_info": {
            "case_id": "CASE_001",
            "incident_date": "2024-01-15",
            "incident_type": "robbery",
            "location": "Main Street Bank"
        },
        "witnesses": [
            {
                "witness_id": "witness_1_john_doe",
                "audio_path": "witness1_statement.mp3",
                "name": "John Doe",
                "role": "customer"
            },
            {
                "witness_id": "witness_2_jane_smith", 
                "transcript": "I was standing outside when I heard shouting...",
                "name": "Jane Smith",
                "role": "passerby"
            },
            {
                "witness_id": "witness_3_officer_jones",
                "audio_path": "officer_statement.wav",
                "name": "Officer Jones",
                "role": "first_responder"
            }
        ]
    }
    
    config_path = "sample_witness_config.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"Sample config created: {config_path}")
    return config_path

def interactive_demo():
    print("INTELLIGENT INTERVIEW ANALYSIS SYSTEM")
    print("="*50)
    
    print("\nSELECT ANALYSIS TYPE:")
    print("1. Single Witness Analysis")
    print("2. Multiple Witness Analysis (generates individual + summary files)")
    print("3. Create Sample Config File")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == '1':
        print("\nSELECT INPUT TYPE:")
        print("1. Audio File") 
        print("2. JSON File")
        print("3. Text Input")
        
        sub_choice = input("\nSelect option (1/2/3): ").strip()
        
        if sub_choice == '1':
            audio_path = input("Enter path to audio file: ").strip()
            if not os.path.exists(audio_path):
                print(f"ERROR: Audio file not found: {audio_path}")
                return
            
            analyzer = WitnessReportAnalyzer()
            results = analyzer.analyze_interview(audio_path=audio_path)
            
            individual_file = analyzer.generate_individual_witness_json(results, "witness")
            results["individual_file"] = individual_file
            
            analyzer.print_analysis_report(results)
            
        elif sub_choice == '2':
            json_path = input("Enter path to JSON file: ").strip()
            transcript = load_transcript_from_json(json_path)
            if transcript:
                analyzer = WitnessReportAnalyzer()
                results = analyzer.analyze_interview(transcript_text=transcript)
                
                individual_file = analyzer.generate_individual_witness_json(results, "witness")
                results["individual_file"] = individual_file
                
                analyzer.print_analysis_report(results)
            
        elif sub_choice == '3':
            print("Enter transcript (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "" and len(lines) > 0 and lines[-1] == "":
                    break
                lines.append(line)
            
            transcript = '\n'.join(lines).strip()
            if len(transcript) > 50:
                analyzer = WitnessReportAnalyzer()
                results = analyzer.analyze_interview(transcript_text=transcript)
                
                individual_file = analyzer.generate_individual_witness_json(results, "witness")
                results["individual_file"] = individual_file
                
                analyzer.print_analysis_report(results)
            else:
                print("ERROR: Transcript too short")
    
    elif choice == '2':
        multi_witness_demo()
    
    elif choice == '3':
        create_sample_witness_config()
        print("\nYou can now use this config file with:")
        print("   python witness_analysis.py --witness-config sample_witness_config.json")
    
    else:
        print("ERROR: Invalid selection")

load_dotenv()

class WitnessReportAnalyzer:
    def __init__(self, openai_api_key: str = None):
     
        self.whisper_model = None
        self.sentiment_classifier = None
        self.nli_classifier = None
        self.empath_analyzer = None
        self.openai_client = None
       
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                self.openai_client.models.list()
            except Exception as e:
                self.openai_client = None
        
        self._initialize_models()

    def main():
        parser = argparse.ArgumentParser(description="Intelligent Interview Analysis System")
        
        parser.add_argument("--audio", help="Path to audio file")
        parser.add_argument("--json", help="Path to JSON file containing transcript")
        parser.add_argument("--transcript", help="Path to transcript text file")
        parser.add_argument("--text", help="Direct transcript text")
        
        parser.add_argument("--witness-config", help="Path to JSON config file with multiple witnesses")
        parser.add_argument("--witness-folder", help="Folder containing multiple audio files")
        
        parser.add_argument("--use-api", action="store_true", help="Use OpenAI API for transcription")
        parser.add_argument("--model-size", default="base",
                            choices=["tiny", "base", "small", "medium", "large"],
                            help="Whisper model size (default: base)")
        parser.add_argument("--output", help="Output JSON file path for single witness")
        parser.add_argument("--summary-output", default="witness_summary.json", 
                            help="Output summary file for multi-witness analysis")
    
        args = parser.parse_args()
        
        analyzer = WitnessReportAnalyzer()
        
        if args.witness_config:
            print("Processing multi-witness configuration...")
            try:
                with open(args.witness_config, 'r') as f:
                    witness_config = json.load(f)
                
                witness_inputs = witness_config.get('witnesses', [])
                if isinstance(witness_config, list) and len(witness_config) > 0 and 'text' in witness_config[0]:
                    witness_inputs = []
                    for i, statement in enumerate(witness_config, 1):
                        witness_inputs.append({
                            'witness_id': f"witness_{i}",
                            'transcript': statement.get('text', '')
                        })
                
                results = analyzer.analyze_multiple_witnesses(
                    witness_inputs=witness_inputs,
                    use_api=args.use_api,
                    model_size=args.model_size
                )
                
                print("Multi-witness analysis complete!")
                print(f"Individual witness files: {len(results.get('individual_files', []))}")
                print(f"Summary report: {results.get('summary_file')}")
                
            except Exception as e:
                print(f"ERROR: Error processing witness config: {e}")
            return
        
        if args.witness_folder:
            print(f"Processing witness folder: {args.witness_folder}")
            try:
                import glob
                audio_files = []
                for ext in ['*.mp3', '*.wav', '*.m4a', '*.flac']:
                    audio_files.extend(glob.glob(os.path.join(args.witness_folder, ext)))
                
                witness_inputs = []
                for i, audio_file in enumerate(audio_files, 1):
                    witness_inputs.append({
                        'witness_id': f"witness_{i}_{os.path.splitext(os.path.basename(audio_file))[0]}",
                        'audio_path': audio_file
                    })
                
                if not witness_inputs:
                    print(f"ERROR: No audio files found in {args.witness_folder}")
                    return
                
                results = analyzer.analyze_multiple_witnesses(
                    witness_inputs=witness_inputs,
                    use_api=args.use_api,
                    model_size=args.model_size
                )
                
                print("Multi-witness analysis complete!")
                print(f"Individual witness files: {len(results.get('individual_files', []))}")
                print(f"Summary report: {results.get('summary_file')}")
                
            except Exception as e:
                print(f"ERROR: Error processing witness folder: {e}")
            return
        
        transcript_text = None
        if args.text:
            transcript_text = args.text
        elif args.transcript:
            with open(args.transcript, 'r') as f:
                transcript_text = f.read()
        elif args.json:
            transcript_text = load_transcript_from_json(args.json)
            if not transcript_text:
                print("ERROR: Could not load transcript from JSON file")
                return
    
        results = analyzer.analyze_interview(
            audio_path=args.audio,
            transcript_text=transcript_text,
            use_api=args.use_api,
            model_size=args.model_size
        )
    
        if transcript_text or args.audio:
            witness_id = "single_witness"
            individual_file = analyzer.generate_individual_witness_json(results, witness_id)
            results["individual_file"] = individual_file
    
        analyzer.print_analysis_report(results)
    
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Analysis saved to: {args.output}")

    def _initialize_models(self):
       
        try:
            self.whisper_model = whisper.load_model("small")
        except Exception as e:
            pass
       
        try:
            self.sentiment_classifier = pipeline("sentiment-analysis")
        except Exception as e:
            pass
       
        try:
            self.nli_classifier = pipeline("text-classification", model="facebook/bart-large-mnli")
        except Exception as e:
            pass
       
        try:
            self.empath_analyzer = Empath()
        except Exception as e:
            pass
       
        print("Models loaded successfully")

    def transcribe_audio(self, audio_path: str, use_api: bool = False, model_size: str = "base") -> Dict[str, Any]:
   
        print(f"Transcribing audio: {os.path.basename(audio_path)}")
       
        if not os.path.exists(audio_path):
            return None
           
        file_size = os.path.getsize(audio_path) / (1024 * 1024)
        
        supported_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac']
        file_ext = os.path.splitext(audio_path)[1].lower()
       
        if file_ext not in supported_formats:
            print(f"WARNING: Unsupported format: {file_ext}")
       
        if use_api and self.openai_client:
            try:
                print("Processing with OpenAI API...")
                with open(audio_path, "rb") as audio_file:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json"
                    )
               
                return {
                    "text": transcript.text,
                    "segments": getattr(transcript, 'segments', []),
                    "language": getattr(transcript, 'language', 'unknown'),
                    "duration": getattr(transcript, 'duration', 0),
                    "method": "api",
                    "model": "whisper-1"
                }
            except Exception as e:
                print("API failed, using local model...")
                use_api = False
       
        if not use_api:
            try:
                if not self.whisper_model or (hasattr(self, '_model_size') and self._model_size != model_size):
                    print(f"Loading Whisper model ({model_size})...")
                    self.whisper_model = whisper.load_model(model_size)
                    self._model_size = model_size
               
                print("Transcribing audio...")
               
                result = self.whisper_model.transcribe(
                    audio_path,
                    verbose=False
                )
               
                return {
                    "text": result["text"],
                    "segments": result["segments"],
                    "language": result.get("language", "unknown"),
                    "method": "local",
                    "model": model_size,
                    "word_count": len(result["text"].split())
                }
            except Exception as e:
                return None

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
       
        if not self.sentiment_classifier:
            return {"error": "Sentiment classifier not available"}
       
        try:
            max_length = 512
            if len(text.split()) > max_length:
                chunks = [text[i:i+max_length*4] for i in range(0, len(text), max_length*4)]
                results = []
                for chunk in chunks:
                    result = self.sentiment_classifier(chunk)
                    results.append(result[0])
               
                positive_scores = [r['score'] for r in results if r['label'] == 'POSITIVE']
                negative_scores = [r['score'] for r in results if r['label'] == 'NEGATIVE']
               
                if positive_scores:
                    avg_positive = sum(positive_scores) / len(positive_scores)
                    return {"label": "POSITIVE", "score": avg_positive, "chunks_analyzed": len(chunks)}
                else:
                    avg_negative = sum(negative_scores) / len(negative_scores)
                    return {"label": "NEGATIVE", "score": avg_negative, "chunks_analyzed": len(chunks)}
            else:
                result = self.sentiment_classifier(text)
                return result[0]
        except Exception as e:
            return {"error": f"Sentiment analysis failed: {e}"}

    def detect_contradictions_with_gpt(self, text: str) -> List[Dict[str, Any]]:

        if not self.openai_client:
            return [{"error": "OpenAI API not available"}]
        
        try:
            statements = self._extract_meaningful_statements(text)
            
            if len(statements) < 2:
                return []
            
            numbered_statements = []
            for i, stmt in enumerate(statements, 1):
                numbered_statements.append(f"{i}. {stmt}")
            
            statements_text = "\n".join(numbered_statements)
            
            prompt = f"""
            Analyze the following interview statements for contradictions. Look for statements that directly contradict each other about facts, timelines, locations, actions, or claims.

            STATEMENTS:
            {statements_text}

            Find pairs of statements that contradict each other. Look for:
            - Contradictory facts (I did vs I didn't)  
            - Timeline contradictions (before vs after, morning vs evening)
            - Location contradictions (here vs there, home vs work)
            - Contradictory claims (always vs never, yes vs no)
            - Presence contradictions (was there vs wasn't there)

            For each ACTUAL contradiction found, return in this exact JSON format:
            [
                {{
                    "statement_a": "exact statement text from the list",
                    "statement_b": "exact contradicting statement text", 
                    "explanation": "brief explanation of why they contradict",
                    "confidence": 85
                }}
            ]

            Return ONLY actual contradictions, not just different topics or opinions. If no contradictions found, return empty array [].
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            
            gpt_response = response.choices[0].message.content.strip()
            
            try:
                import json
                if gpt_response.startswith('[') and gpt_response.endswith(']'):
                    contradictions_data = json.loads(gpt_response)
                else:
                    start = gpt_response.find('[')
                    end = gpt_response.rfind(']') + 1
                    if start >= 0 and end > start:
                        json_part = gpt_response[start:end]
                        contradictions_data = json.loads(json_part)
                    else:
                        contradictions_data = []
                
                if not isinstance(contradictions_data, list):
                    contradictions_data = []
                
                contradictions = []
                for i, item in enumerate(contradictions_data):
                    if isinstance(item, dict):
                        contradictions.append({
                            "pair_id": i + 1,
                            "premise": item.get("statement_a", ""),
                            "hypothesis": item.get("statement_b", ""), 
                            "explanation": item.get("explanation", ""),
                            "label": "CONTRADICTION",
                            "confidence": min(item.get("confidence", 85) / 100.0, 1.0),
                            "is_contradiction": True,
                            "method": "gpt_analysis"
                        })
                
                return contradictions
                
            except json.JSONDecodeError as e:
                contradictions = []
                if "contradiction" in gpt_response.lower() or "contradict" in gpt_response.lower():
                    contradictions.append({
                        "pair_id": 1,
                        "premise": "GPT detected contradictions but response format was unclear",
                        "hypothesis": "See raw GPT response in logs", 
                        "explanation": "GPT found contradictions but JSON parsing failed",
                        "label": "CONTRADICTION",
                        "confidence": 0.7,
                        "is_contradiction": True,
                        "method": "gpt_analysis_fallback",
                        "raw_response": gpt_response
                    })
                
                return contradictions
                
        except Exception as e:
            return [{"error": f"GPT analysis failed: {e}"}]

    def _extract_meaningful_statements(self, text: str) -> List[str]:
        import re
        
        sentences = re.split(r'[.!?]+', text)
        
        meaningful_statements = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            if (len(sentence) > 25 and
                any(word in sentence.lower() for word in 
                    ['i', 'she', 'he', 'we', 'they', 'was', 'were', 'did', 'didn\'t', 
                     'never', 'always', 'yes', 'no', 'before', 'after', 'went', 'saw', 'told', 'said']) and
                not sentence.lower().startswith(('um', 'uh', 'well', 'like', 'you know', 'so', 'and')) and
                sentence.count(' ') >= 3):
                
                meaningful_statements.append(sentence)
        
        return meaningful_statements[:15]

    def detect_contradictions(self, statements: List[Tuple[str, str]] = None, text: str = None) -> List[Dict[str, Any]]:

        if text and self.openai_client:
            return self.detect_contradictions_with_gpt(text)
        
        if statements and self.nli_classifier:
            return self._detect_contradictions_original(statements)
        
        return []

    def _detect_contradictions_original(self, statements: List[Tuple[str, str]]) -> List[Dict[str, Any]]:

        if not self.nli_classifier:
            return [{"error": "NLI classifier not available"}]
       
        if not statements:
            return []
       
        results = []
        contradictions_found = 0
        
        for i, (premise, hypothesis) in enumerate(statements):
            try:
                premise = premise.strip()
                hypothesis = hypothesis.strip()
                
                if len(premise) < 10 or len(hypothesis) < 10:
                    continue
                
                input_text = f"{premise} </s></s> {hypothesis}"
                
                result = self.nli_classifier(input_text)
                
                label = result[0]["label"]
                confidence = result[0]["score"]
                is_contradiction = label.upper() == "CONTRADICTION"
                
                if is_contradiction:
                    contradictions_found += 1
                
                pair_result = {
                    "pair_id": i + 1,
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": label,
                    "confidence": confidence,
                    "is_contradiction": is_contradiction,
                    "method": "nli_fallback"
                }
                
                results.append(pair_result)
                    
            except Exception as e:
                results.append({
                    "pair_id": i + 1,
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "error": f"Analysis failed: {e}",
                    "is_contradiction": False
                })
        
        return results

    def analyze_psychological_categories(self, text: str) -> Dict[str, float]:

        if not self.empath_analyzer:
            return {"error": "Empath analyzer not available"}
       
        try:
            scores = self.empath_analyzer.analyze(text, normalize=True)
           
            relevant_categories = [
                'negative_emotion', 'positive_emotion', 'anxiety', 'anger',
                'sadness', 'fear', 'confusion', 'confidence', 'aggression',
                'nervousness', 'trust', 'deception', 'honesty'
            ]
           
            filtered_scores = {cat: scores.get(cat, 0.0) for cat in relevant_categories if cat in scores}
           
            top_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
           
            return {
                "relevant_categories": filtered_scores,
                "top_categories": dict(top_categories),
                "total_categories": len(scores)
            }
        except Exception as e:
            return {"error": f"Psychological analysis failed: {e}"}

    def generate_followup_questions(self, transcript: str, analysis_results: Dict[str, Any]) -> List[str]:

        if not self.openai_client:
            return ["OpenAI API not available - set OPENAI_API_KEY for follow-up questions"]
       
        try:
            contradictions = analysis_results.get('contradictions', [])
            if contradictions and isinstance(contradictions, list):
                contradictions_count = len([r for r in contradictions if isinstance(r, dict) and r.get('is_contradiction')])
            else:
                contradictions_count = 0
            
            psychological = analysis_results.get('psychological', {})
            if isinstance(psychological, dict) and 'relevant_categories' in psychological:
                psych_indicators = list(psychological['relevant_categories'].keys())[:5]
            else:
                psych_indicators = []
            
            context = f"""
            Interview Transcript: {transcript[:1000]}...
           
            Sentiment Analysis: {analysis_results.get('sentiment', 'N/A')}
            Contradictions Found: {contradictions_count}
            Psychological Indicators: {psych_indicators}
            """
           
            prompt = f"""
            As an expert interviewer, analyze this interview data and generate 5 intelligent follow-up questions
            that would help clarify inconsistencies, probe emotional stress points, or uncover additional information.
           
            {context}
           
            Generate questions that are:
            1. Specific and targeted
            2. Non-confrontational but probing
            3. Designed to reveal inconsistencies or emotional responses
            4. Professional and appropriate for a formal interview
           
            Return only the 5 questions, numbered 1-5.
            """
           
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            if response and response.choices and len(response.choices) > 0:
                questions_text = response.choices[0].message.content
                
                if questions_text:
                    lines = questions_text.strip().split('\n')
                    questions = []
                    
                    for line in lines:
                        line = line.strip()
                        if line and (line[0].isdigit() or line.startswith(('1.', '2.', '3.', '4.', '5.'))):
                            question = line
                            if '.' in question[:5]:
                                question = question.split('.', 1)[1].strip()
                            questions.append(question)
                    
                    if questions:
                        return questions[:5]
                    else:
                        sentences = [s.strip() + '?' for s in questions_text.split('?') if len(s.strip()) > 10]
                        return sentences[:5]
                else:
                    return ["No response content received from OpenAI API"]
            else:
                return ["Invalid response structure from OpenAI API"]
           
        except Exception as e:
            return [
                "Can you clarify your earlier statement about the timeline of events?",
                "How did you feel when that situation occurred?", 
                "Are there any details you might have overlooked in your initial response?",
                "What was your immediate reaction to what happened?",
                "Is there anything else you think might be relevant to mention?"
            ]

    def _extract_potential_contradictions(self, text: str) -> List[Tuple[str, str]]:

        import re
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        contradiction_pairs = []
        
        contradiction_patterns = [
            (r'\b(yes|yeah|definitely|absolutely|of course|for sure)\b', r'\b(no|never|not|didn\'t|wasn\'t|couldn\'t)\b'),
            (r'\b(before|earlier|first|initially|morning)\b', r'\b(after|later|last|finally|evening|night)\b'),
            (r'\b(inside|indoors|home|office)\b', r'\b(outside|outdoors|away|street)\b'),
            (r'\b(here|nearby|close)\b', r'\b(there|far|distant|away)\b'),
            (r'\b(I was|I went|I saw|I met|I talked)\b', r'\b(I wasn\'t|I didn\'t go|I didn\'t see|I never met|I never talked)\b'),
            (r'\b(I know|I remember|I recall|I\'m sure)\b', r'\b(I don\'t know|I don\'t remember|I forget|I\'m not sure|I have no idea)\b'),
            (r'\b(many|lots|several|multiple|all|everyone)\b', r'\b(few|none|nobody|no one|nothing)\b'),
            (r'\b(I did|I called|I sent|I went|I bought)\b', r'\b(I didn\'t|I never called|I never sent|I never went|I never bought)\b')
        ]
        
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences):
                if i >= j or abs(j - i) < 2:
                    continue
                
                sent1_lower = sent1.lower()
                sent2_lower = sent2.lower()
                
                for pos_pattern, neg_pattern in contradiction_patterns:
                    if (re.search(pos_pattern, sent1_lower) and re.search(neg_pattern, sent2_lower)) or \
                       (re.search(neg_pattern, sent1_lower) and re.search(pos_pattern, sent2_lower)):
                        contradiction_pairs.append((sent1, sent2))
                        break
                
                if len(contradiction_pairs) >= 10:
                    break
            
            if len(contradiction_pairs) >= 10:
                break
        
        if len(contradiction_pairs) < 3:
            key_subjects = ['time', 'when', 'where', 'who', 'what', 'how', 'meeting', 'call', 'email', 'document', 'person', 'place']
            
            for subject in key_subjects:
                subject_sentences = [s for s in sentences if subject in s.lower() and len(s) > 30]
                
                for i in range(len(subject_sentences)):
                    for j in range(i + 2, len(subject_sentences)):
                        if len(contradiction_pairs) < 8:
                            contradiction_pairs.append((subject_sentences[i], subject_sentences[j]))
                
                if len(contradiction_pairs) >= 8:
                    break
        
        return contradiction_pairs[:8]

    def extract_crime_facts_with_gpt(self, transcript: str, witness_id: str = None) -> Dict[str, Any]:

        if not self.openai_client:
            return {"error": "OpenAI API not available"}
        
        try:
            prompt = f"""
            You are analyzing a witness statement about a crime. Extract ONLY the factual details about what happened. Focus on:

            WITNESS STATEMENT:
            {transcript}

            Extract the following information in JSON format. Use "unknown" if information is not mentioned:

            {{
                "witness_id": "{witness_id or 'unknown'}",
                "crime_type": "what type of crime occurred",
                "perpetrator": {{
                    "description": "physical description",
                    "clothing": "what they wore", 
                    "behavior": "how they acted",
                    "weapon": "weapon used if any"
                }},
                "victim": {{
                    "description": "victim description",
                    "condition": "injured/unharmed/unknown",
                    "response": "how victim reacted"
                }},
                "crime_details": {{
                    "what_happened": "detailed description of the crime",
                    "sequence_of_events": ["event 1", "event 2", "event 3"],
                    "duration": "how long it took",
                    "method": "how the crime was committed"
                }},
                "location": {{
                    "address": "specific location",
                    "type": "indoor/outdoor/vehicle/etc",
                    "surroundings": "what was around",
                    "lighting": "bright/dark/etc"
                }},
                "time": {{
                    "date": "date if mentioned",
                    "time": "time of day",
                    "duration": "how long witness observed"
                }},
                "witness_position": {{
                    "distance": "how far away",
                    "view_quality": "clear/obstructed/partial",
                    "vantage_point": "where witness was standing/sitting"
                }},
                "evidence": {{
                    "physical": "items left behind, damage",
                    "sounds": "what witness heard",
                    "other_witnesses": "other people present"
                }},
                "confidence_level": "how certain the witness seems (high/medium/low)"
            }}

            Return ONLY valid JSON. Be precise and factual."""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            
            gpt_response = response.choices[0].message.content.strip()
            
            try:
                if gpt_response.startswith('{') and gpt_response.endswith('}'):
                    crime_facts = json.loads(gpt_response)
                else:
                    start = gpt_response.find('{')
                    end = gpt_response.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_part = gpt_response[start:end]
                        crime_facts = json.loads(json_part)
                    else:
                        return {"error": "Could not parse crime facts from GPT response"}
                
                return crime_facts
                
            except json.JSONDecodeError as e:
                return {
                    "error": f"JSON parsing failed: {e}",
                    "raw_response": gpt_response,
                    "witness_id": witness_id or "unknown"
                }
                
        except Exception as e:
            return {"error": f"Crime fact extraction failed: {e}"}

    def generate_individual_witness_json(self, witness_data: Dict[str, Any], witness_id: str = None) -> str:
        if not witness_id:
            witness_id = witness_data.get('witness_id', 'witness_unknown')
        
        safe_witness_id = re.sub(r'[^\w\-_]', '_', witness_id)
        output_path = f"{safe_witness_id}.json"
        
        transcript_text = ""
        if witness_data.get('transcript'):
            transcript_text = witness_data['transcript'].get('text', '')
        
        crime_facts = {}
        if transcript_text:
            crime_facts = self.extract_crime_facts_with_gpt(transcript_text, witness_id)
        
        contradictions = witness_data.get('contradictions', [])
        contradiction_count = len([r for r in contradictions if isinstance(r, dict) and r.get('is_contradiction', False)])
        
        sentiment = witness_data.get('sentiment', {})
        sentiment_summary = "unknown"
        if isinstance(sentiment, dict) and 'label' in sentiment:
            confidence = sentiment.get('score', 0)
            sentiment_summary = f"{sentiment['label'].lower()} ({confidence:.0%} confidence)"
        
        individual_witness_report = {
            "witness_info": {
                "witness_id": witness_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "transcript_length": len(transcript_text),
                "analysis_status": "complete" if transcript_text else "no_transcript"
            },
            "crime_facts": crime_facts,
            "credibility_assessment": {
                "contradictions_found": contradiction_count,
                "emotional_state": sentiment_summary,
                "confidence_level": crime_facts.get('confidence_level', 'unknown')
            },
            "follow_up_questions": witness_data.get('followup_questions', []),
            "raw_transcript": transcript_text if len(transcript_text) < 2000 else transcript_text[:2000] + "... [truncated]"
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(individual_witness_report, f, indent=2, ensure_ascii=False)
        
        print(f"Individual witness report saved: {output_path}")
        return output_path

    def cross_verify_witness_reports(self, witness_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.openai_client:
            return {"error": "OpenAI API not available"}
        
        try:
            witnesses_text = ""
            for i, report in enumerate(witness_reports, 1):
                if "error" not in report:
                    witnesses_text += f"\nWITNESS {i} ({report.get('witness_id', 'unknown')}):\n"
                    witnesses_text += json.dumps(report, indent=2)
                    witnesses_text += "\n" + "="*50
            
            if not witnesses_text:
                return {"error": "No valid witness reports to cross-verify"}
            
            prompt = f"""
            You are analyzing multiple witness statements about the same crime. Your job is to:
            1. Find facts that are CONFIRMED by multiple witnesses (similar or identical details)
            2. Identify CONFLICTING information between witnesses
            3. Extract the most reliable crime narrative

            WITNESS REPORTS:
            {witnesses_text}

            Analyze and return in this JSON format:

            {{
                "confirmed_facts": {{
                    "crime_type": "confirmed type of crime",
                    "perpetrator": {{
                        "description": "details confirmed by multiple witnesses",
                        "clothing": "clothing details multiple witnesses agree on",
                        "weapon": "weapon if confirmed by multiple witnesses",
                        "behavior": "behaviors confirmed by multiple witnesses"
                    }},
                    "victim": {{
                        "description": "confirmed victim details",
                        "condition": "confirmed victim condition",
                        "response": "confirmed victim response"
                    }},
                    "crime_sequence": {{
                        "what_happened": "step-by-step confirmed sequence",
                        "method": "confirmed method of crime",
                        "duration": "confirmed timeline"
                    }},
                    "location": {{
                        "address": "confirmed location",
                        "type": "confirmed location type",
                        "surroundings": "confirmed environmental details",
                        "lighting": "confirmed lighting conditions"
                    }},
                    "time": {{
                        "date": "confirmed date",
                        "time": "confirmed time",
                        "duration": "confirmed observation duration"
                    }},
                    "evidence": {{
                        "physical": "confirmed physical evidence",
                        "sounds": "confirmed sounds heard",
                        "other_people": "confirmed other people present"
                    }}
                }},
                "conflicting_information": [
                    {{
                        "category": "what aspect differs",
                        "witness_1": "first version",
                        "witness_2": "conflicting version",
                        "severity": "high/medium/low"
                    }}
                ],
                "reliability_assessment": {{
                    "most_reliable_witness": "witness ID with most consistent details",
                    "least_reliable_witness": "witness ID with most inconsistencies",
                    "overall_confidence": "high/medium/low confidence in the facts"
                }},
                "crime_summary": "Brief summary of what definitely happened based on confirmed facts only"
            }}

            Only include details in confirmed_facts if they appear in multiple witness reports or are stated with high confidence by a single reliable witness. Use 'unknown' for unconfirmed details."""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            gpt_response = response.choices[0].message.content.strip()
            
            try:
                if gpt_response.startswith('{') and gpt_response.endswith('}'):
                    verification_results = json.loads(gpt_response)
                else:
                    start = gpt_response.find('{')
                    end = gpt_response.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_part = gpt_response[start:end]
                        verification_results = json.loads(json_part)
                    else:
                        return {"error": "Could not parse cross-verification results"}
                
                return verification_results
                
            except json.JSONDecodeError as e:
                return {
                    "error": f"JSON parsing failed: {e}",
                    "raw_response": gpt_response
                }
                
        except Exception as e:
            return {"error": f"Cross-verification failed: {e}"}

    def generate_simplified_summary_json(self, witness_data: List[Dict[str, Any]], output_path: str = "witness_summary.json") -> str:
        print("Generating witness summary...")
        
        witness_facts = []
        simplified_witnesses = []
        
        for i, witness in enumerate(witness_data, 1):
            if witness.get('transcript'):
                transcript_text = witness['transcript'].get('text', '')
                if transcript_text:
                    witness_id = witness.get('witness_id', f"witness_{i}")
                    facts = self.extract_crime_facts_with_gpt(transcript_text, witness_id)
                    witness_facts.append(facts)
                    
                    contradictions = witness.get('contradictions', [])
                    contradiction_count = len([r for r in contradictions if isinstance(r, dict) and r.get('is_contradiction', False)])
                    
                    sentiment = witness.get('sentiment', {})
                    emotional_state = "neutral"
                    if isinstance(sentiment, dict) and 'label' in sentiment:
                        emotional_state = sentiment['label'].lower()
                    
                    simplified_witnesses.append({
                        "witness_id": witness_id,
                        "credibility": {
                            "contradictions_found": contradiction_count,
                            "emotional_state": emotional_state,
                            "reliability": "high" if contradiction_count == 0 else "medium" if contradiction_count < 3 else "low"
                        },
                        "key_observations": {
                            "crime_type": facts.get('crime_type', 'unknown'),
                            "perpetrator_description": facts.get('perpetrator', {}).get('description', 'unknown'),
                            "location": facts.get('location', {}).get('address', 'unknown'),
                            "time_of_incident": facts.get('time', {}).get('time', 'unknown')
                        }
                    })
        
        if not witness_facts:
            print("ERROR: No valid witness transcripts found")
            return None
        
        cross_verification = self.cross_verify_witness_reports(witness_facts)
        
        confirmed_facts = cross_verification.get('confirmed_facts', {})
        
        forensic_summary = {
            "case_summary": {
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "total_witnesses": len(witness_facts),
                "reliability_overview": cross_verification.get('reliability_assessment', {}).get('overall_confidence', 'unknown'),
                "incident_summary": cross_verification.get('crime_summary', 'Unable to determine from available evidence')
            },
            
            "confirmed_evidence": {
                "incident_type": confirmed_facts.get('crime_type', 'unknown'),
                
                "suspect_description": {
                    "physical_appearance": confirmed_facts.get('perpetrator', {}).get('description', 'unknown'),
                    "clothing": confirmed_facts.get('perpetrator', {}).get('clothing', 'unknown'),
                    "weapon_used": confirmed_facts.get('perpetrator', {}).get('weapon', 'none reported'),
                    "behavior": confirmed_facts.get('perpetrator', {}).get('behavior', 'unknown')
                },
                
                "incident_location": {
                    "address": confirmed_facts.get('location', {}).get('address', 'unknown'),
                    "location_type": confirmed_facts.get('location', {}).get('type', 'unknown'),
                    "lighting_conditions": confirmed_facts.get('location', {}).get('lighting', 'unknown')
                },
                
                "timeline": {
                    "date": confirmed_facts.get('time', {}).get('date', 'unknown'),
                    "time": confirmed_facts.get('time', {}).get('time', 'unknown'),
                    "incident_duration": confirmed_facts.get('crime_sequence', {}).get('duration', 'unknown')
                },
                
                "victim_information": {
                    "description": confirmed_facts.get('victim', {}).get('description', 'unknown'),
                    "condition": confirmed_facts.get('victim', {}).get('condition', 'unknown'),
                    "response": confirmed_facts.get('victim', {}).get('response', 'unknown')
                },
                
                "physical_evidence": confirmed_facts.get('evidence', {}).get('physical', 'none reported'),
                
                "sequence_of_events": confirmed_facts.get('crime_sequence', {}).get('what_happened', 'unknown')
            },
            
            "witness_reliability": simplified_witnesses,
            
            "conflicting_accounts": [
                {
                    "issue": conflict.get('category', 'unknown'),
                    "conflicting_details": f"{conflict.get('witness_1', 'N/A')} vs {conflict.get('witness_2', 'N/A')}",
                    "impact": conflict.get('severity', 'unknown')
                }
                for conflict in cross_verification.get('conflicting_information', [])
            ],
            
            "investigation_notes": {
                "most_reliable_witness": cross_verification.get('reliability_assessment', {}).get('most_reliable_witness', 'unknown'),
                "areas_needing_clarification": [conflict.get('category', 'unknown') for conflict in cross_verification.get('conflicting_information', [])],
                "evidence_strength": "strong" if len([w for w in simplified_witnesses if w['credibility']['reliability'] == 'high']) >= 2 else "moderate" if len(simplified_witnesses) >= 2 else "limited"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(forensic_summary, f, indent=2, ensure_ascii=False)
        
        print(f"Witness summary saved: {output_path}")
        return output_path

    def analyze_multiple_witnesses(self, witness_inputs: List[Dict[str, Any]], 
                                 use_api: bool = False, model_size: str = "base") -> Dict[str, Any]:

        print(f"Starting analysis of {len(witness_inputs)} witnesses...")
        
        all_witness_data = []
        individual_files = []
        
        for i, witness_input in enumerate(witness_inputs, 1):
            print(f"Processing witness {i}/{len(witness_inputs)}")
            
            witness_id = witness_input.get('witness_id', f"witness_{i}")
            
            if witness_input.get('audio_path'):
                results = self.analyze_interview(
                    audio_path=witness_input['audio_path'],
                    use_api=use_api,
                    model_size=model_size
                )
            elif witness_input.get('transcript'):
                results = self.analyze_interview(
                    transcript_text=witness_input['transcript']
                )
            else:
                print(f"WARNING: No valid input for witness {i}")
                continue
            
            results['witness_id'] = witness_id
            all_witness_data.append(results)
            
            individual_file = self.generate_individual_witness_json(results, witness_id)
            individual_files.append(individual_file)
        
        if not all_witness_data:
            print("ERROR: No witness data could be analyzed")
            return {"error": "No valid witness data"}
        
        print("Generating cross-verification and summary...")
        summary_file = self.generate_simplified_summary_json(all_witness_data, "witness_summary.json")
        
        print("Multi-witness analysis complete")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_witnesses": len(all_witness_data),
            "individual_analyses": all_witness_data,
            "individual_files": individual_files,
            "summary_file": summary_file,
            "status": "complete"
        }

    def save_raw_analysis_data(self, results: Dict[str, Any], audio_path: str = None, witness_id: str = None) -> str:
        if witness_id:
            safe_witness_id = re.sub(r'[^\w\-_]', '_', witness_id)
            output_filename = f"{safe_witness_id}_raw_analysis.json"
        elif audio_path:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_filename = f"raw_analysis_{base_name}.json"
        else:
            output_filename = f"witness_raw_analysis.json"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        return output_filename

    def analyze_interview(self, audio_path: str = None, transcript_text: str = None,
                          contradiction_pairs: List[Tuple[str, str]] = None,
                          use_api: bool = False, model_size: str = "base") -> Dict[str, Any]:

        results = {
            "timestamp": datetime.now().isoformat(),
            "audio_file": audio_path,
            "transcript": None,
            "sentiment": None,
            "contradictions": None,
            "psychological": None,
            "followup_questions": None
        }
       
        if audio_path and os.path.exists(audio_path):
            transcript_result = self.transcribe_audio(audio_path, use_api=use_api, model_size=model_size)
            if transcript_result:
                results["transcript"] = transcript_result
                transcript_text = transcript_result["text"]
                print(f"Transcription complete ({len(transcript_text)} characters)")
            else:
                print("ERROR: Transcription failed")
                return results
        elif transcript_text:
            print("Using provided transcript")
            results["transcript"] = {"text": transcript_text, "method": "provided"}
        else:
            print("ERROR: No audio file or transcript provided")
            return results
       
        if not contradiction_pairs and transcript_text:
            contradiction_pairs = self._extract_potential_contradictions(transcript_text)
       
        if transcript_text:
            print("Analyzing sentiment...")
            sentiment_result = self.analyze_sentiment(transcript_text)
            results["sentiment"] = sentiment_result
            if "error" not in sentiment_result:
                print(f"Sentiment: {sentiment_result.get('label', 'N/A')} ({sentiment_result.get('score', 0):.1%} confidence)")
            else:
                print("WARNING: Sentiment analysis failed")
       
        print("Analyzing contradictions...")
        if transcript_text:
            contradiction_results = self.detect_contradictions(text=transcript_text)
            results["contradictions"] = contradiction_results
            
            actual_contradictions = [r for r in contradiction_results if isinstance(r, dict) and r.get('is_contradiction', False)]
            if actual_contradictions:
                print(f"Found {len(actual_contradictions)} contradictions")
            else:
                print("No contradictions detected")
        elif contradiction_pairs:
            contradiction_results = self.detect_contradictions(statements=contradiction_pairs)
            results["contradictions"] = contradiction_results
            print("Contradiction analysis complete")
        else:
            results["contradictions"] = []
            print("No contradiction analysis performed")
       
        if transcript_text:
            print("Analyzing psychological indicators...")
            psychological_result = self.analyze_psychological_categories(transcript_text)
            results["psychological"] = psychological_result
            
            if "error" not in psychological_result:
                relevant = psychological_result.get("relevant_categories", {})
                top_indicators = sorted(relevant.items(), key=lambda x: x[1], reverse=True)[:3]
                if top_indicators and any(score > 0 for _, score in top_indicators):
                    print("Psychological analysis complete")
                else:
                    print("No significant psychological indicators detected")
            else:
                print("WARNING: Psychological analysis failed")
       
        if transcript_text:
            print("Generating follow-up questions...")
            followup_questions = self.generate_followup_questions(transcript_text, results)
            results["followup_questions"] = followup_questions
            print(f"Generated {len(followup_questions)} follow-up questions")
       
        if audio_path or transcript_text:
            witness_id = results.get('witness_id', 'witness_unknown')
            raw_data_file = self.save_raw_analysis_data(results, audio_path, witness_id)
            results["raw_data_file"] = raw_data_file
       
        return results

    def print_analysis_report(self, results: Dict[str, Any]):
        print("\n" + "="*80)
        print("INTERVIEW ANALYSIS REPORT")
        print("="*80)
       
        if results.get("transcript"):
            transcript = results["transcript"]["text"]
            print(f"\nTRANSCRIPT ({len(transcript)} characters)")
            print("-" * 40)
            print(transcript)
       
        if results.get("sentiment"):
            sentiment = results["sentiment"]
            print(f"\nSENTIMENT ANALYSIS")
            print("-" * 40)
            if "error" not in sentiment:
                print(f"Overall Sentiment: {sentiment.get('label', 'N/A')}")
                print(f"Confidence: {sentiment.get('score', 0):.2%}")
            else:
                print(f"Error: {sentiment['error']}")
       
        print(f"\nCONTRADICTION ANALYSIS")
        print("-" * 40)
        
        contradictions = results.get("contradictions", [])
        
        if not contradictions:
            print("No contradiction analysis performed.")
        elif isinstance(contradictions, list) and len(contradictions) == 1 and "error" in contradictions[0]:
            print(f"Error: {contradictions[0]['error']}")
        else:
            actual_contradictions = [r for r in contradictions if isinstance(r, dict) and r.get('is_contradiction', False)]
            
            if actual_contradictions:
                print(f"Found {len(actual_contradictions)} contradictions:")
                for i, contradiction in enumerate(actual_contradictions, 1):
                    print(f"\n{i}. Statement A: {contradiction.get('premise', 'N/A')}")
                    print(f"   Statement B: {contradiction.get('hypothesis', 'N/A')}")
                    
                    if 'explanation' in contradiction:
                        print(f"   Explanation: {contradiction['explanation']}")
                    
                    print(f"   Confidence: {contradiction.get('confidence', 0):.1%}")
            else:
                print("No contradictions detected.")
       
        if results.get("psychological"):
            psych = results["psychological"]
            print(f"\nPSYCHOLOGICAL ANALYSIS")
            print("-" * 40)
            if "error" not in psych:
                relevant = psych.get("relevant_categories", {})
                top_categories = sorted(relevant.items(), key=lambda x: x[1], reverse=True)[:5]
               
                print("Top Psychological Indicators:")
                for category, score in top_categories:
                    if score > 0:
                        print(f"   {category.replace('_', ' ').title()}: {score:.3f}")
            else:
                print(f"Error: {psych['error']}")
       
        if results.get("followup_questions"):
            questions = results["followup_questions"]
            print(f"\nSUGGESTED FOLLOW-UP QUESTIONS")
            print("-" * 40)
            for i, question in enumerate(questions, 1):
                print(f"{i}. {question}")
       
        if results.get("individual_files") or results.get("summary_file"):
            print(f"\nFILES GENERATED")
            print("-" * 40)
            if results.get("individual_files"):
                print("Individual witness reports:")
                for file in results["individual_files"]:
                    print(f"  • {file}")
            if results.get("summary_file"):
                print(f"Summary report: {results['summary_file']}")
       
        print("\n" + "="*80)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("INTELLIGENT INTERVIEW ANALYSIS SYSTEM")
        print("="*75)
        print("\nUSAGE OPTIONS:")
        
        print("\nSINGLE WITNESS ANALYSIS:")
        print("   python witness_analysis.py --audio interview.mp3")
        print("   python witness_analysis.py --json transcript.json")
        print("   python witness_analysis.py --transcript transcript.txt")
        print("   python witness_analysis.py --text \"Your transcript text here\"")
        
        print("\nMULTIPLE WITNESS ANALYSIS:")
        print("   python witness_analysis.py --witness-folder /path/to/audio/files/")
        print("   python witness_analysis.py --witness-config witness_config.json")
        
        print("\nADVANCED OPTIONS:")
        print("   python witness_analysis.py --audio interview.wav --model-size medium")
        print("   python witness_analysis.py --witness-folder audio/ --use-api")
        print("   python witness_analysis.py --witness-config config.json --summary-output final_report.json")
       
        print("\nAPI KEY SETUP:")
        print("   Set environment variable: OPENAI_API_KEY=your_key_here")
        print("   Required for: GPT analysis, contradiction detection, witness cross-verification")
       
        print("\nSUPPORTED INPUT FORMATS:")
        print("   Audio: MP3, WAV, M4A, FLAC, OGG, WMA, AAC")
        print("   JSON: Various transcript formats")
        print("   Text: Plain text files or direct input")
        
        print("\nWITNESS CONFIG JSON FORMAT:")
        print('   {')
        print('     "witnesses": [')
        print('       {"witness_id": "witness_1", "audio_path": "audio1.mp3"},')
        print('       {"witness_id": "witness_2", "transcript": "text here"}')
        print('     ]')
        print('   }')
       
        print("\nWHISPER MODEL SIZES:")
        print("   tiny  - ~39 MB, fastest")
        print("   base  - ~74 MB, recommended")  
        print("   small - ~244 MB, more accurate")
        print("   medium - ~769 MB, high accuracy")
        print("   large  - ~1550 MB, best accuracy")
        
        print("\nOUTPUT FILES:")
        print("   Individual witness files: witness_1.json, witness_2.json, etc.")
        print("   witness_summary.json - Simplified forensic summary")
        print("   *_raw_analysis.json - Detailed technical analysis")
       
        print("\nKEY FEATURES:")
        print("   Separate JSON file for each witness")
        print("   Simplified summary focused on forensic essentials")
        print("   Human-readable format for investigators")
        print("   Enhanced credibility assessment")
       
        print("\n" + "="*75)
        demo_choice = input("Run interactive demo? (y/n): ").strip().lower()
        if demo_choice in ['y', 'yes']:
            interactive_demo()
        else:
            print("\nRun with --witness-folder or --witness-config for multi-witness analysis")
            print("Run with --audio, --json, or --text for single witness analysis")
    else:
        main()
            
