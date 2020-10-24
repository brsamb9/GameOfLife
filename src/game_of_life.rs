/*
Game of life by Horton Conway:
    More details: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life

Top-level description

# Create an initial randomised state
    grid of n x n matrix consisting of live and dead cells in a 'state'.
    The next state is dependent on the previous by the governing four rules.

# loops throughout states
    features in game:
        move mouse to create a light-weight spaceship in a random direction

# 'Esc' button to quit

*/
extern crate rand;
use std::collections::HashMap;

// Create a grid
#[derive(Debug)]
pub struct Ensemble(pub Vec<Option<State>>);

impl Ensemble {
    pub fn new(size: usize, number_of_states_to_hold: usize) -> Self {
        // Fill previous states with None to initialise
        let mut all_states: Vec<Option<State>> = Vec::with_capacity(number_of_states_to_hold);
        for _ in 0..number_of_states_to_hold - 1 {
            all_states.push(None);
        }

        let mut initial_state: Vec<u32> = Vec::with_capacity(size);

        for _ in 0..size {
            let random_number: u32 = rand::random();
            initial_state.push(random_number % 2);
        }

        all_states.push(Some(State(initial_state)));

        Ensemble(all_states)
    }

    pub fn next_state(&mut self) {
        // Pops out oldest state 
        self.0.remove(0); 
        // Push next state

        let current_state = self.num_states() - 1;
        let new_state = match &self.0[current_state] {
            Some(i) => Some(State::state_creation(&i)), // pushes next state -> always at the capacity
            None => None,
        };
        
        self.0.push(new_state);
    }

    pub fn apply_life(&mut self, index: usize) {
        if index < self.size_of_state() {
            match &mut self.0[4] {
                Some(ok) => ok.0[index] = 1,
                None => (),
            }
        }
    }
    pub fn size_of_state(&self) -> usize {
        match &self.0[self.num_states() - 1] {
            Some(i) => i.0.len(),
            None => 0,
        }
    }

    fn num_states(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct State(pub Vec<u32>);
impl State {
    pub fn state_creation(&self) -> Self {
        let total_length = self.0.len();
        let mut next_state: Vec<u32> = Vec::with_capacity(total_length);

        // Input (N) to program is always > N*N
        let n = (total_length as f64).sqrt() as usize;

        for indx in 0..total_length {
            // Grab current cell
            let row = indx / n;
            let col = indx % n;
            
            // Find number of live cells surrounding cell - Careful about PBC
            let surrounding_life: u32 = State::neighbours(row, col, n).iter().map(|i| self.0[*i]).sum();
            
            let curr_cell = self.0[row * n + col];
            let new_cell: u32 = State::apply_rules(curr_cell, surrounding_life);
            next_state.push(new_cell);
        }
        State(next_state)
    }
    
    fn neighbours(row: usize, col: usize, n: usize) -> [usize; 8] {
        let i_minus = |i: usize| {
            if i == 0 { n - 1 } else { i - 1}
        };
        let i_plus = |i: usize| {
            if i == n - 1 { 0 } else { i + 1}
        };
        [   (i_minus(row) * n) + i_minus(col),
            (i_minus(row) * n) + col,
            (i_minus(row) * n) + i_plus(col),
            (row * n) + i_minus(col),
            (row * n) + i_plus(col),
            (i_plus(row) * n) + i_minus(col),
            (i_plus(row) * n) + col,
            (i_plus(row) * n) + i_plus(col),
        ]
    }

    fn apply_rules(curr_cell: u32, life: u32) -> u32 {
        match curr_cell {
            1 => {
                match life {
                    0 | 1 => 0, // 1)  Any live cell with fewer than two live neighbours dies, as if by underpopulation.
                    2 | 3 => 1, // 2)  Any live cell with two or three live neighbours lives on to the next generation.
                    _ => 0, // 3)  Any live cell with more than three live neighbours dies, as if by overpopulation.
                }
            }
            0 => {
                // 4)  Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
                match life {
                    3 => 1,
                    _ => 0,
                }
            }
            _ => panic!("Neither dead or alive!! I've found schrodinger's cat!"),
        }
    }
}


#[derive(Debug, Eq, PartialEq, Hash)]
pub enum Glider {
    LightWeightSpaceShip,
}
#[derive(Debug, Eq, PartialEq)]
pub struct RandomGlider{
    dir: u32,
    glider: Glider,
    pub map: Vec<u32>,
}

impl RandomGlider {
    pub fn assign(glider: Glider, n: u32, mouse_position: [f64; 2]) -> Self {
        let mut dir: u32 = rand::random();
        dir = dir % 4;
        let map = RandomGlider::get_map(&glider, dir, n, mouse_position);

        RandomGlider {
            dir, // 0: left (as given), 1: down, 2: right, 3: up (clockwise)
            glider,
            map,
        }
    }

    fn get_map(g: &Glider, dir: u32, n: u32, mouse_position: [f64; 2]) -> Vec<u32> {
        // Could make this static, but don't think it's worth it -> via lazy_static
        let mut relative_coords: HashMap<Glider, Vec<(i32, i32)>> = HashMap::new();
        // From 'middle / upper middle' position -> relative coordinate (row, col) of live cells
        relative_coords.insert(
            // Glider::LightWeightSpaceShip, vec![(-1, -1), (-1, 2), (0, -2), (1, -2), (1, 2), (2, -2), (2, -1), (2, 0), (2, 1)]
            Glider::LightWeightSpaceShip, vec![(-1, -1), (-1, 2), (0, -2), (1, -2), (1, 2), (2, -2), (2, -1), (2, 0), (2, 1)]
        );

         // rotate if needed
        let ship_coords: Vec<(i32, i32)> = relative_coords[g].clone().iter().map(|xy| 
            match dir {
                0 => (xy.0, xy.1),
                1 => (-xy.1, xy.0),
                2 => (-xy.0, -xy.1),
                3 => (xy.1, -xy.0),
                _ => panic!("Invalid direction")
            }
        ).collect();

        // given as [col, row] -> as u32 will truncate value
        let (mouse_row, mouse_col) = (mouse_position[1] as i32, mouse_position[0] as i32);
        // let mouse_index = (mouse_row * n as i32) +  mouse_col;

        // Collapse into 1d - extension of neighbour function, don't want to mess with it though
        let ship_coord_to_index = |coord: (i32, i32), n: i32| {
            // Edge cases such as -ve overflow, but not important enough to convolute code with the fix
            ((mouse_row + coord.0) * n + (mouse_col + coord.1)) as u32
        };
        ship_coords.iter().map(|xy| ship_coord_to_index(*xy, n as i32)).collect()
    }
}



#[cfg(test)]
mod test {

    use super::*;
    #[test]
    fn test_glider(){
        let mut answer = RandomGlider::get_map(&Glider::LightWeightSpaceShip, 0, 500, [5.0, 5.0]);
        answer.sort();
        let expected = vec![2004, 2007, 2503, 3003, 3007, 3503, 3504, 3505, 3506];
        assert!(answer == expected);
    }
    #[test]
    fn test_output_iterator() {
        const SIZE: usize = 49;
        let mut s = Ensemble::new(SIZE, 5);

        let mut length = s.0[4].clone().unwrap().0.len();
        let mut prev_state = s.0[4].clone().unwrap().0;
        for i in 0..=10000 { 
            s.next_state(); 

            let check_if_all_dead: u32 = prev_state.iter().sum();
            if check_if_all_dead  == 0 {
                s = Ensemble::new(SIZE, 5);

                length = s.0[4].clone().unwrap().0.len();
                prev_state = s.0[4].clone().unwrap().0;
            }

            if i % 1000 == 0 {
                let curr = s.0[4].clone().unwrap();
                assert!(prev_state != curr.0);
                assert!(length == curr.0.len());
                prev_state = curr.0;
            }
        }
    }

    #[test]
    fn test_next_state() {
        const SIZE: usize = 49;
        let mut s = Ensemble::new(SIZE, 5);
        assert!(SIZE == s.size_of_state());
        assert!(5 == s.num_states());

        // one extra state from initial state
        for i in 0..5 { 
            s.next_state(); 
            assert!(s.0[4-i].is_some());
        }

        let to_pop_out = s.0[0].clone();
        let curr_last = s.0[4].clone();
        s.next_state();
        // Actually popped out
        assert!(to_pop_out != s.0[0].clone());
        // And placed into the latest spot
        assert!(curr_last != s.0[4].clone());
        // Which was pushed down            
        assert!(curr_last.unwrap().0 == s.0[3].clone().unwrap().0);
    }

    #[test]
    fn state_test(){
        let s = State(vec![
            0,0,0,0,0,0,0,
            0,0,1,0,0,1,0,
            0,1,0,0,0,0,0,
            0,1,0,0,0,1,0,
            0,1,1,1,1,0,0,
            0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,
            ]);
        // let expected_answers: Vec<u32> = vec![4, 4, 5, 3, 5, 4, 3, 4, 4, 4, 5, 3, 6, 3, 4, 3];
        // Driver function test
        let expected_next_state = State(vec![
            0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,
            0,1,1,0,0,0,0,
            1,1,0,1,1,0,0,
            0,1,1,1,1,0,0,
            0,0,1,1,0,0,0,
            0,0,0,0,0,0,0,
            ]);

        let test = s.state_creation().0;

        println!("\nActual:");
        for i in 14..7*6 {
            print!("{} ", test[i]);
        }
        println!("\nExpected:");
        for i in 14..7*6 {
            print!("{} ", expected_next_state.0[i]);          
        }
        println!("\n");


        // assert!(test == expected_next_state.0);
    }
    #[test]
    fn life_result_test() {
        let curr_life = 1;
        assert!(State::apply_rules(curr_life, 0) == 0 ); // rule 1
        assert!(State::apply_rules(curr_life, 1) == 0 ); // rule 1
        assert!(State::apply_rules(curr_life, 2) == 1 ); // rule 2
        assert!(State::apply_rules(curr_life, 3) == 1 ); // rule 2
        assert!(State::apply_rules(curr_life, 4) == 0 ); // rule 3

        let curr_life = 0;
        assert!(State::apply_rules(curr_life, 0) == 0 ); // rule 3
        assert!(State::apply_rules(curr_life, 1) == 0 ); // rule 3
        assert!(State::apply_rules(curr_life, 2) == 0 ); // rule 3
        assert!(State::apply_rules(curr_life, 3) == 1 ); // rule 3
        assert!(State::apply_rules(curr_life, 4) == 0 ); // rule 3
    }
    #[test]
    fn surround_life_check() {
        const SIZE: usize = 4;
        let s = State(vec![1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1]);
        let expected_answers: Vec<u32> = vec![4, 4, 5, 3, 5, 4, 3, 4, 4, 4, 5, 3, 6, 3, 4, 3];
        assert!(s.0.len() == SIZE*SIZE && expected_answers.len() == SIZE*SIZE); // to avoid typos in test

        // functions with main driver function
        let mut life_answers: Vec<u32> = Vec::new();
        for index in 0..SIZE*SIZE {
            let row = index / SIZE;
            let col = index % SIZE; 
            
            life_answers.push(State::neighbours(row, col, 4).iter().map(|i| s.0[*i]).sum());
        }
        for (a, b) in life_answers.iter().zip(expected_answers.iter()) {
            assert!(a == b);
        }
        
    }
    #[test]
    fn neighbour_test() {
        const SIZE: usize = 4;

        let expected_neighbours: Vec<[usize; 8]> = vec![
            // row 1
            [1, 3, 4, 5, 7, 12, 13, 15], // i.e. 0 index -> these indexes are its neighbour
            [0, 2, 4, 5, 6, 12, 13, 14],
            [1, 3, 5, 6, 7, 13, 14, 15],
            [0, 2, 4, 6, 7, 12, 14, 15],
            // row 2
            [0, 1, 3, 5, 7, 8, 9, 11],
            [0, 1, 2, 4, 6, 8, 9, 10],
            [1, 2, 3, 5, 7, 9, 10, 11],
            [0, 2, 3, 4, 6, 8, 10, 11],
            // row 3
            [4, 5, 7, 9, 11, 12, 13, 15],
            [4, 5, 6, 8, 10, 12, 13, 14],
            [5, 6, 7, 9, 11, 13, 14, 15],
            [4, 6, 7, 8, 10, 12, 14, 15],
            // row 4
            [0, 1, 3, 8, 9, 11, 13, 15],
            [0, 1, 2, 8, 9, 10, 12, 14],
            [1, 2, 3, 9, 10, 11, 13, 15],
            [0, 2, 3, 8, 10, 11, 12, 14],
        ];

        for index in 0..SIZE*SIZE {
            let row = index / SIZE;
            let col = index % SIZE; 
            let mut test = State::neighbours(row, col, SIZE);
            test.sort();
            assert!(test == expected_neighbours[index]);
        }


    }
}
