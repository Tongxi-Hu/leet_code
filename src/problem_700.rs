use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::common::TreeNode;

/// p605
pub fn can_place_flowers(flowerbed: Vec<i32>, n: i32) -> bool {
    let m = flowerbed.len();
    let mut n = n;
    let mut i = 0;
    while i < m {
        if (i == 0 || flowerbed[i - 1] == 0)
            && flowerbed[i] == 0
            && (i == m - 1 || flowerbed[i + 1] == 0)
        {
            n -= 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    n <= 0
}

/// p606
pub fn tree2str(root: Option<Rc<RefCell<TreeNode>>>) -> String {
    fn dfs(root: &Option<Rc<RefCell<TreeNode>>>) -> Option<String> {
        if let Some(node) = root.as_ref() {
            let mut res = String::new();
            let v = node.borrow().val;
            res = res + &v.to_string();
            let left_res = dfs(&node.borrow().left);
            let right_res = dfs(&node.borrow().right);
            if let Some(left_val) = left_res.as_ref() {
                res = res + "(";
                res = res + &left_val.to_string();
                res = res + ")";
            }
            if let Some(right_val) = right_res {
                if left_res == None {
                    res = res + "()";
                }
                res = res + "(";
                res = res + &right_val.to_string();
                res = res + ")";
            }
            return Some(res);
        } else {
            None
        }
    }
    dfs(&root).unwrap_or("".to_string())
}

/// p609
pub fn find_duplicate(paths: Vec<String>) -> Vec<Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();

    for path in paths.into_iter() {
        let vals = path.split(' ').collect::<Vec<_>>();
        for &s in vals.iter().skip(1) {
            let idx = s.find('(').unwrap();
            let cur_path = vals[0].to_string() + "/" + &s[..idx];
            let cur_content = s[idx..].to_string();
            map.entry(cur_content).or_insert(Vec::new()).push(cur_path);
        }
    }

    map.into_values()
        .filter(|vals| vals.len() > 1)
        .collect::<Vec<_>>()
}

/// p611
pub fn triangle_number(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    nums.sort();
    let length = nums.len();
    if length < 3 {
        return 0;
    }
    let mut count = 0;
    for i in 0..length - 2 {
        for j in i + 1..length - 1 {
            for k in j + 1..length {
                if nums[i] + nums[j] > nums[k] {
                    count = count + 1;
                }
            }
        }
    }
    count
}

/// p617
pub fn merge_trees(
    root1: Option<Rc<RefCell<TreeNode>>>,
    root2: Option<Rc<RefCell<TreeNode>>>,
) -> Option<Rc<RefCell<TreeNode>>> {
    fn dfs(
        root1: &Option<Rc<RefCell<TreeNode>>>,
        root2: &Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        match (root1.as_ref(), root2.as_ref()) {
            (None, None) => None,
            (Some(node1), Some(node2)) => {
                let mut node = TreeNode::new(node1.borrow().val + node2.borrow().val);
                node.left = dfs(&node1.borrow().left, &node2.borrow().left);
                node.right = dfs(&node1.borrow().right, &node2.borrow().right);
                Some(Rc::new(RefCell::new(node)))
            }
            (Some(node1), None) => Some(node1.clone()),
            (None, Some(node2)) => Some(node2.clone()),
        }
    }
    dfs(&root1, &root2)
}

/// p621
pub fn least_interval(tasks: Vec<char>, n: i32) -> i32 {
    let mut counts = [0; 26];
    let mut max_count = 0;
    let mut count = 0;

    for idx in tasks.iter().map(|&task| ((task as u8) - b'A') as usize) {
        counts[idx] += 1;
        if counts[idx] > max_count {
            count = 1;
            max_count = counts[idx];
        } else if counts[idx] == max_count {
            count += 1;
        }
    }

    i32::max(tasks.len() as i32, count + (max_count - 1) * (n + 1))
}

/// p622
struct MyCircularQueue {
    arr: Vec<i32>,
    rear: usize,
    curr_size: usize,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl MyCircularQueue {
    fn new(k: i32) -> Self {
        MyCircularQueue {
            arr: vec![0; k as usize],
            rear: 0,
            curr_size: 0,
        }
    }

    fn en_queue(&mut self, value: i32) -> bool {
        if self.is_full() {
            false
        } else {
            self.arr[self.rear] = value;
            self.curr_size += 1;
            self.rear = (self.rear + 1) % self.arr.len();
            true
        }
    }

    fn de_queue(&mut self) -> bool {
        if self.is_empty() {
            false
        } else {
            self.curr_size -= 1;
            true
        }
    }

    fn front(&self) -> i32 {
        if self.is_empty() {
            -1
        } else {
            self.arr[(self.rear - self.curr_size + self.arr.len()) % self.arr.len()]
        }
    }

    fn rear(&self) -> i32 {
        if self.is_empty() {
            -1
        } else {
            self.arr[(self.rear - 1 + self.arr.len()) % self.arr.len()]
        }
    }

    fn is_empty(&self) -> bool {
        self.curr_size == 0
    }

    fn is_full(&self) -> bool {
        self.curr_size == self.arr.len()
    }
}

///p623
pub fn add_one_row(
    root: Option<Rc<RefCell<TreeNode>>>,
    val: i32,
    depth: i32,
) -> Option<Rc<RefCell<TreeNode>>> {
    use std::collections::VecDeque;
    if depth == 1 {
        let mut node = TreeNode::new(val);
        node.left = root;
        return Some(Rc::new(RefCell::new(node)));
    }
    let mut queue = VecDeque::new();
    queue.push_back(root.clone());
    for i in 0..depth - 2 {
        let size = queue.len();
        for j in 0..size {
            if let Some(node) = queue.pop_front() {
                if node.as_ref().unwrap().borrow().left.is_some() {
                    queue.push_back(node.as_ref().unwrap().borrow().left.clone());
                }
                if node.as_ref().unwrap().borrow().right.is_some() {
                    queue.push_back(node.as_ref().unwrap().borrow().right.clone());
                }
            }
        }
    }
    while !queue.is_empty() {
        if let Some(node) = queue.pop_front() {
            let mut new_node = TreeNode::new(val);
            new_node.left = node.as_ref().unwrap().borrow().left.clone();
            node.as_ref().unwrap().borrow_mut().left = Some(Rc::new(RefCell::new(new_node)));
            let mut new_node = TreeNode::new(val);
            new_node.right = node.as_ref().unwrap().borrow().right.clone();
            node.as_ref().unwrap().borrow_mut().right = Some(Rc::new(RefCell::new(new_node)));
        }
    }
    root
}

/// p624
pub fn max_distance(arrays: Vec<Vec<i32>>) -> i32 {
    let mut res = 0;
    let mut min_val = arrays[0][0];
    let mut max_val = arrays[0][arrays[0].len() - 1];
    for i in 1..arrays.len() {
        let n = arrays[i].len();
        res = res.max((arrays[i][n - 1] - min_val).abs());
        res = res.max((max_val - arrays[i][0]).abs());
        min_val = min_val.min(arrays[i][0]);
        max_val = max_val.max(arrays[i][n - 1]);
    }
    res
}
