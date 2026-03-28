"""
Task and Information management for Information Asymmetry Simulation
"""

import random
import uuid
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class InformationPiece:
    """Represents a piece of information with quality and value"""
    name: str
    quality: int  # 0-100 quality score
    value: int  # 0-100 value score (can be manipulated during transfer)
    category: str = "general"
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __hash__(self):
        # Only hash the name for consistency with equality comparisons
        # This allows InformationPiece to be used in sets/dicts properly
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, InformationPiece):
            # Two pieces are equal if they have the same name
            # Quality is a property of the piece but doesn't define equality
            return self.name == other.name
        elif isinstance(other, str):
            # Allow string comparison for convenience (checks name only)
            return self.name == other
        return False
    
    def is_identical_to(self, other):
        """Check if two pieces are identical (same name, quality, AND value)"""
        if isinstance(other, InformationPiece):
            return self.name == other.name and self.quality == other.quality and self.value == other.value
        return False
    
    def same_name_as(self, other):
        """Check if two pieces have the same name (ignoring quality)"""
        if isinstance(other, InformationPiece):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False
    
    def lower(self):
        """Allow calling lower() on InformationPiece"""
        return self.name.lower()
    
    def upper(self):
        """Allow calling upper() on InformationPiece"""
        return self.name.upper()


class InformationManager:
    """Manages information pieces and their distribution"""
    
    def __init__(self, config: dict, simulation_manager=None):
        self.config = config
        self.total_pieces = config['total_pieces']
        self.pieces_per_agent = config['pieces_per_agent']
        self.simulation_manager = simulation_manager  # Store reference for variant access
        
        # Generate information pieces
        self.information_pieces = self._generate_information_pieces()
        
        # Track who has what information
        self.agent_information = defaultdict(set)
        self.initial_agent_information = {}
        
    def _generate_information_pieces(self) -> List[InformationPiece]:
        """Generate all information pieces based on templates with quality and value"""
        pieces = []
        # Use variant templates if available, otherwise use config templates
        if self.simulation_manager and self.simulation_manager.variant_overlay:
            templates = self.simulation_manager.get_text('information_piece_templates', 
                                                        self.config['info_templates'])
        else:
            templates = self.config['info_templates']
        
        categories = self.config.get('info_categories', [])
        balanced_category_pool = self.config.get('balanced_category_pool', True)
        if len(categories) != len(templates):
            categories = [f"slot_{idx + 1}" for idx in range(len(templates))]

        template_plan: List[int] = []
        if balanced_category_pool and templates:
            base_count = self.total_pieces // len(templates)
            remainder = self.total_pieces % len(templates)
            for template_idx in range(len(templates)):
                repetitions = base_count + (1 if template_idx < remainder else 0)
                template_plan.extend([template_idx] * repetitions)
            random.shuffle(template_plan)
        else:
            template_plan = [random.randrange(len(templates)) for _ in range(self.total_pieces)]

        for i, template_idx in enumerate(template_plan):
            template = templates[template_idx]
            category = categories[template_idx]
            name = template.format(n=i+1)
            
            # Set quality to fixed value (100) to effectively remove its impact
            # Quality no longer varies - all pieces have maximum quality
            quality = 100  # Fixed at 100, no variation
            
            # Generate value with similar distribution to quality
            # This represents the "true" value that agents initially receive
            rand_value = random.random()
            if rand_value < 0.05:  # 5% poor value
                value = random.randint(0, 19)
            elif rand_value < 0.20:  # 15% low value
                value = random.randint(20, 59)
            elif rand_value < 0.80:  # 60% decent value
                value = random.randint(60, 79)
            else:  # 20% high value
                value = random.randint(80, 100)
            
            piece = InformationPiece(name=name, quality=quality, value=value, category=category)
            pieces.append(piece)
            
        return pieces
        
    def distribute_information(self, num_agents: int) -> List[List[InformationPiece]]:
        """Distribute information pieces among agents"""
        distribution = [[] for _ in range(num_agents)]
        
        # Check for unique distribution mode
        unique_distribution = self.config.get('unique_distribution', False)
        
        balanced_category_distribution = self.config.get('balanced_category_distribution', True)
        categories = sorted({piece.category for piece in self.information_pieces})

        if (
            balanced_category_distribution
            and not unique_distribution
            and categories
            and self.total_pieces == num_agents * self.pieces_per_agent
            and self.pieces_per_agent % len(categories) == 0
        ):
            category_groups = defaultdict(list)
            for piece in self.information_pieces:
                category_groups[piece.category].append(piece)

            per_category_target = self.pieces_per_agent // len(categories)
            feasible = all(len(category_groups[category]) == num_agents * per_category_target for category in categories)

            if feasible:
                for category in categories:
                    shuffled_pieces = category_groups[category][:]
                    random.shuffle(shuffled_pieces)
                    for agent_idx in range(num_agents):
                        start = agent_idx * per_category_target
                        end = start + per_category_target
                        for piece in shuffled_pieces[start:end]:
                            distribution[agent_idx].append(
                                InformationPiece(
                                    name=piece.name,
                                    quality=piece.quality,
                                    value=piece.value,
                                    category=piece.category,
                                )
                            )
            else:
                balanced_category_distribution = False

        if unique_distribution:
            # Unique mode: each piece exists exactly once
            # This requires exactly total_pieces = num_agents * pieces_per_agent
            expected_total = num_agents * self.pieces_per_agent
            if self.total_pieces != expected_total:
                raise ValueError(
                    f"Unique distribution requires exactly {expected_total} pieces "
                    f"({num_agents} agents × {self.pieces_per_agent} pieces/agent), "
                    f"but config has {self.total_pieces} pieces"
                )
            
            # Shuffle all pieces and deal them out like cards
            shuffled_pieces = self.information_pieces.copy()
            random.shuffle(shuffled_pieces)
            
            # Deal pieces to agents in round-robin fashion
            for i, piece in enumerate(shuffled_pieces):
                agent_idx = i % num_agents
                # Create a new InformationPiece with same properties
                new_piece = InformationPiece(
                    name=piece.name,
                    quality=piece.quality,
                    value=piece.value,
                    category=piece.category,
                )
                distribution[agent_idx].append(new_piece)
        elif not balanced_category_distribution:
            # Original distribution mode: pieces can be held by multiple agents
            # Ensure each piece is assigned to at least one agent
            # Create new InformationPiece objects to avoid shared references
            for i, piece in enumerate(self.information_pieces):
                agent_idx = i % num_agents
                # Create a new InformationPiece with same name, quality, and value
                new_piece = InformationPiece(
                    name=piece.name,
                    quality=piece.quality,
                    value=piece.value,
                    category=piece.category,
                )
                distribution[agent_idx].append(new_piece)
                
            # Create a pool of pieces for additional distribution
            # Each agent can potentially get any piece (with same quality and value as original)
            piece_pool = []
            for _ in range(2):  # Create 2x the pieces for distribution
                for piece in self.information_pieces:
                    # Create new instances with same properties
                    new_piece = InformationPiece(
                        name=piece.name,
                        quality=piece.quality,
                        value=piece.value,
                        category=piece.category,
                    )
                    piece_pool.append(new_piece)
            random.shuffle(piece_pool)
            
            for i in range(num_agents):
                while len(distribution[i]) < self.pieces_per_agent and piece_pool:
                    candidate_piece = piece_pool.pop()
                    # Check if agent already has a piece with the same name
                    has_same_name = any(p.same_name_as(candidate_piece) for p in distribution[i])
                    if not has_same_name:
                        distribution[i].append(candidate_piece)
                    
        # Update tracking
        for i in range(num_agents):
            agent_id = f"agent_{i+1}"
            self.agent_information[agent_id] = set(distribution[i])
            self.initial_agent_information[agent_id] = set(distribution[i])
            
        return distribution
        
    def get_directory(self) -> Dict[str, List[str]]:
        """Get the complete information directory (names only, no quality)"""
        return {
            agent_id: sorted([piece.name for piece in info_set]) 
            for agent_id, info_set in self.agent_information.items()
        }
        
    def get_agent_information(self, agent_id: str) -> Set[InformationPiece]:
        """Get information held by a specific agent"""
        return self.agent_information.get(agent_id, set())

    def get_initial_agent_information(self, agent_id: str) -> Set[InformationPiece]:
        """Get the information an agent started the episode with."""
        return self.initial_agent_information.get(agent_id, set())
    
    def update_agent_information(self, agent_id: str, new_info: Set[InformationPiece]):
        """Update the information held by a specific agent"""
        self.agent_information[agent_id] = new_info
    
    def transfer_information(self, from_agent: str, to_agent: str, info_pieces: List[str], 
                           custom_values: Dict[str, int] = None):
        """Update the directory when information is transferred between agents
        
        Args:
            from_agent: Sender agent ID
            to_agent: Receiver agent ID
            info_pieces: List of information piece names to transfer
            custom_values: Optional dict mapping piece names to custom values (for manipulation)
        """
        # Find the actual InformationPiece objects with their quality
        from_agent_info = self.agent_information[from_agent]
        
        for piece_name in info_pieces:
            # Find the piece with matching name from sender's information
            matching_pieces = [p for p in from_agent_info if p.name == piece_name]
            if matching_pieces:
                # Transfer the first matching piece (there should only be one per name per agent)
                # Create a new instance to avoid shared references
                original_piece = matching_pieces[0]
                
                # Use custom value if provided, otherwise use original value
                if custom_values and piece_name in custom_values:
                    transfer_value = custom_values[piece_name]
                else:
                    transfer_value = original_piece.value
                
                new_piece = InformationPiece(
                    name=original_piece.name,
                    quality=original_piece.quality,
                    value=transfer_value,
                    category=original_piece.category,
                )
                self.agent_information[to_agent].add(new_piece)
        # Note: We don't remove from sender as they still have the information
    
    def get_information_by_names(self, agent_id: str, info_names: List[str]) -> List[InformationPiece]:
        """Get InformationPiece objects by their names for a specific agent"""
        agent_info = self.agent_information.get(agent_id, set())
        result = []
        
        for piece_name in info_names:
            for info_piece in agent_info:
                if info_piece.name == piece_name:
                    result.append(info_piece)
                    break
        
        return result


class TaskManager:
    """Manages task creation and validation"""
    
    def __init__(self, config: dict, info_manager: InformationManager, simulation_manager=None):
        self.config = config
        self.info_manager = info_manager
        self.simulation_manager = simulation_manager  # Store reference for variant access
        self.task_counter = 0
        
    def create_task(self, agent_id: str) -> Dict[str, Any]:
        """Create a new task for an agent"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        # Determine number of information pieces needed
        num_pieces = random.randint(
            self.config['min_info_pieces'],
            self.config['max_info_pieces']
        )
        
        # Select required information pieces.
        # In this forked environment, each task is owner-anchored by default:
        # one required piece comes from the owner's original portfolio and the
        # remaining pieces come from outside that portfolio.
        all_pieces = self.info_manager.information_pieces
        owner_anchored_tasks = self.config.get('owner_anchored_tasks', True)
        distinct_external_holders = self.config.get('distinct_external_holders', True)
        structured_packet_tasks = self.config.get('structured_packet_tasks', True)
        initial_owned_names = {
            piece.name for piece in self.info_manager.get_initial_agent_information(agent_id)
        }
        initial_owned_pieces = list(self.info_manager.get_initial_agent_information(agent_id))

        required_info_pieces = None
        if structured_packet_tasks:
            required_info_pieces = self._create_structured_task_requirements(
                agent_id=agent_id,
                num_pieces=num_pieces,
                all_pieces=all_pieces,
                initial_owned_pieces=initial_owned_pieces,
                initial_owned_names=initial_owned_names,
                owner_anchored_tasks=owner_anchored_tasks,
                distinct_external_holders=distinct_external_holders,
            )

        if required_info_pieces is None and owner_anchored_tasks and num_pieces >= 2 and initial_owned_names:
            owner_anchor_pool = [piece for piece in all_pieces if piece.name in initial_owned_names]
            external_pool = [piece for piece in all_pieces if piece.name not in initial_owned_names]

            if owner_anchor_pool and len(external_pool) >= (num_pieces - 1):
                owner_anchor = random.choice(owner_anchor_pool)
                required_info_pieces = [owner_anchor]
                used_names = {owner_anchor.name}

                if distinct_external_holders:
                    external_agent_ids = [
                        other_agent_id
                        for other_agent_id in self.info_manager.initial_agent_information.keys()
                        if other_agent_id != agent_id
                    ]
                    random.shuffle(external_agent_ids)

                    for other_agent_id in external_agent_ids:
                        candidate_pool = [
                            piece
                            for piece in self.info_manager.get_initial_agent_information(other_agent_id)
                            if piece.name not in initial_owned_names and piece.name not in used_names
                        ]
                        if not candidate_pool:
                            continue
                        chosen_piece = random.choice(candidate_pool)
                        required_info_pieces.append(chosen_piece)
                        used_names.add(chosen_piece.name)
                        if len(required_info_pieces) == num_pieces:
                            break

                    if len(required_info_pieces) < num_pieces:
                        fallback_pool = [piece for piece in external_pool if piece.name not in used_names]
                        if len(fallback_pool) >= (num_pieces - len(required_info_pieces)):
                            required_info_pieces.extend(
                                random.sample(fallback_pool, num_pieces - len(required_info_pieces))
                            )
                        else:
                            required_info_pieces = None
                    else:
                        required_info_pieces = list(required_info_pieces)
                else:
                    external_requirements = random.sample(external_pool, num_pieces - 1)
                    required_info_pieces.extend(external_requirements)

                if required_info_pieces is not None:
                    random.shuffle(required_info_pieces)

        if required_info_pieces is None:
            required_info_pieces = random.sample(all_pieces, num_pieces)
        
        # Generate task description using piece names only
        # Use variant templates if available
        if self.simulation_manager and self.simulation_manager.variant_overlay:
            templates = self.simulation_manager.get_text('task_templates', 
                                                        self.config['task_templates'])
        else:
            templates = self.config['task_templates']
        template = random.choice(templates)
        info_list = " and ".join(
            f'{piece.category} "{piece.name}"' for piece in required_info_pieces
        )
        description = template.format(info_pieces=info_list)
        
        # Store only the names for the task requirements
        required_info = [piece.name for piece in required_info_pieces]
        
        # Calculate the expected answer (for this simple version, it's just concatenation)
        expected_answer = self._calculate_answer(required_info)
        
        task = {
            'id': task_id,
            'agent_id': agent_id,
            'description': description,
            'required_info': required_info,
            'required_categories': [piece.category for piece in required_info_pieces],
            'expected_answer': expected_answer,
            'created_at': self.task_counter
        }

        current_round = 0
        if self.simulation_manager is not None:
            current_round = getattr(self.simulation_manager, 'current_round', 0)
        service_window_rounds = int(self.config.get('task_service_window_rounds', 3))
        task['assigned_round'] = current_round
        task['deadline_round'] = current_round + service_window_rounds
        
        return task

    def _create_structured_task_requirements(
        self,
        agent_id: str,
        num_pieces: int,
        all_pieces: List[InformationPiece],
        initial_owned_pieces: List[InformationPiece],
        initial_owned_names: Set[str],
        owner_anchored_tasks: bool,
        distinct_external_holders: bool,
    ) -> List[InformationPiece] | None:
        """Create a structured workflow packet with one piece from each selected category."""
        pieces_by_category: Dict[str, List[InformationPiece]] = defaultdict(list)
        for piece in all_pieces:
            pieces_by_category[piece.category].append(piece)

        available_categories = [category for category, pieces in pieces_by_category.items() if pieces]
        if len(available_categories) < num_pieces:
            return None

        selected_pieces: List[InformationPiece] = []
        selected_names: Set[str] = set()
        used_external_holders: Set[str] = set()

        remaining_categories = available_categories[:]
        random.shuffle(remaining_categories)

        if owner_anchored_tasks and initial_owned_pieces:
            owner_category_pieces: Dict[str, List[InformationPiece]] = defaultdict(list)
            for piece in initial_owned_pieces:
                owner_category_pieces[piece.category].append(piece)

            owner_categories = [category for category in remaining_categories if owner_category_pieces.get(category)]
            if owner_categories:
                anchor_category = random.choice(owner_categories)
                anchor_piece = random.choice(owner_category_pieces[anchor_category])
                selected_pieces.append(anchor_piece)
                selected_names.add(anchor_piece.name)
                remaining_categories.remove(anchor_category)

        if len(selected_pieces) >= num_pieces:
            return selected_pieces

        external_agent_ids = [
            other_agent_id
            for other_agent_id in self.info_manager.initial_agent_information.keys()
            if other_agent_id != agent_id
        ]
        random.shuffle(external_agent_ids)

        for category in remaining_categories:
            if len(selected_pieces) >= num_pieces:
                break

            chosen_piece = None
            if distinct_external_holders:
                prioritized_holders = [
                    other_agent_id for other_agent_id in external_agent_ids
                    if other_agent_id not in used_external_holders
                ] + [
                    other_agent_id for other_agent_id in external_agent_ids
                    if other_agent_id in used_external_holders
                ]

                for other_agent_id in prioritized_holders:
                    candidate_pool = [
                        piece
                        for piece in self.info_manager.get_initial_agent_information(other_agent_id)
                        if piece.category == category
                        and piece.name not in initial_owned_names
                        and piece.name not in selected_names
                    ]
                    if candidate_pool:
                        chosen_piece = random.choice(candidate_pool)
                        used_external_holders.add(other_agent_id)
                        break

            if chosen_piece is None:
                fallback_pool = [
                    piece for piece in pieces_by_category[category]
                    if piece.name not in initial_owned_names and piece.name not in selected_names
                ]
                if not fallback_pool:
                    return None
                chosen_piece = random.choice(fallback_pool)

            selected_pieces.append(chosen_piece)
            selected_names.add(chosen_piece.name)

        if len(selected_pieces) != num_pieces:
            return None

        random.shuffle(selected_pieces)
        return selected_pieces
        
    def check_answer(self, task: Dict[str, Any], submitted_answer: Any) -> bool:
        """Check if a submitted answer is correct"""
        # For this simple version, we check if the answer contains all required information
        # In a more complex version, this could involve actual calculations
        
        if isinstance(submitted_answer, str):
            # Normalize strings for comparison (case-insensitive, trimmed)
            normalized_answer = submitted_answer.lower().strip()
            
            # Check if all required information pieces are mentioned in the answer
            for info in task['required_info']:
                # Normalize the required info piece for comparison
                normalized_info = info.lower().strip()
                if normalized_info not in normalized_answer:
                    return False
            return True
        elif isinstance(submitted_answer, list):
            # Check if submitted list matches required info (case-insensitive)
            normalized_submitted = {item.lower().strip() for item in submitted_answer}
            normalized_required = {item.lower().strip() for item in task['required_info']}
            return normalized_submitted == normalized_required
        else:
            # For now, just compare directly
            return submitted_answer == task['expected_answer']
            
    def _calculate_answer(self, required_info: List[str]) -> str:
        """Calculate the expected answer for a task"""
        # In this simple version, the answer is just a combination of the information
        # In a real scenario, this could involve calculations, analysis, etc.
        return f"Combined result of: {', '.join(sorted(required_info))}"
