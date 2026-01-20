use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    rc::Rc,
    sync::{
        Condvar, Mutex,
        mpsc::{Receiver, Sender, channel},
    },
};

use crate::common::TreeNode;

/// 1103
pub fn distribute_candies(mut candies: i32, n: i32) -> Vec<i32> {
    let n = n as usize;
    let mut ans = vec![0; n];
    let mut i = 1;
    while candies > 0 {
        ans[(i - 1) as usize % n] += i.min(candies);
        candies -= i;
        i += 1;
    }
    ans
}

/// 1104
pub fn path_in_zig_zag_tree(label: i32) -> Vec<i32> {
    let mut x = 2_i32.pow((label as f64).log2() as u32);
    let mut label = label;
    let mut ret = vec![label];

    while label > 1 {
        label = x - 1 - ((label / 2) % (x / 2));
        x /= 2;
        ret.push(label);
    }

    ret.reverse();
    ret
}

/// 1105
pub fn min_height_shelves(books: Vec<Vec<i32>>, shelf_width: i32) -> i32 {
    let n = books.len();
    let mut dp = vec![0; n + 1];
    for i in 1..=n {
        let (mut j, mut width, mut max_height) = (i - 1, books[i - 1][0], books[i - 1][1]);
        dp[i] = dp[i - 1] + max_height;
        while j > 0 && width + books[j - 1][0] <= shelf_width {
            max_height = max_height.max(books[j - 1][1]);
            width += books[j - 1][0];
            dp[i] = dp[i].min(dp[j - 1] + max_height);
            j -= 1;
        }
    }
    dp[n]
}

/// 1106
pub fn parse_bool_expr(expression: String) -> bool {
    let mut stack = VecDeque::new();
    expression.chars().for_each(|c| match c {
        '(' | 't' | 'f' | '!' | '&' | '|' => {
            stack.push_back(c);
        }
        ')' => {
            let mut list = vec![];
            while *stack.back().unwrap() != '(' {
                list.push(stack.pop_back().unwrap())
            }
            stack.pop_back();
            match stack.pop_back().unwrap() {
                '&' => {
                    if list.contains(&'f') {
                        stack.push_back('f')
                    } else {
                        stack.push_back('t')
                    }
                }
                '!' => {
                    if list.contains(&'f') {
                        stack.push_back('t')
                    } else {
                        stack.push_back('f')
                    }
                }
                '|' => {
                    if list.contains(&'t') {
                        stack.push_back('t')
                    } else {
                        stack.push_back('f')
                    }
                }
                _ => (),
            }
        }
        ',' => (),
        _ => (),
    });
    stack.pop_back().unwrap() == 't'
}

/// 1108
pub fn defang_i_paddr(address: String) -> String {
    address.replace(".", "[.]")
}

/// 1109
pub fn corp_flight_bookings(bookings: Vec<Vec<i32>>, n: i32) -> Vec<i32> {
    let (mut get_on, mut get_off) = (HashMap::new(), HashMap::new());
    bookings.iter().for_each(|b| {
        let on = get_on.entry(b[0]).or_insert(0);
        *on = *on + b[2];
        let off = get_off.entry(b[1] + 1).or_insert(0);
        *off = *off + b[2];
    });
    let (mut total, mut time_line) = (0, vec![0; n as usize]);
    time_line.iter_mut().enumerate().for_each(|(day, cnt)| {
        let (on, off) = (
            get_on.get(&((day + 1) as i32)).unwrap_or(&0),
            get_off.get(&((day + 1) as i32)).unwrap_or(&0),
        );
        total = total + on - off;
        *cnt = total;
    });
    time_line
}

/// 1110
pub fn del_nodes(
    root: Option<Rc<RefCell<TreeNode>>>,
    mut to_delete: Vec<i32>,
) -> Vec<Option<Rc<RefCell<TreeNode>>>> {
    let mut ret = vec![];
    fn dfs(
        root: Option<Rc<RefCell<TreeNode>>>,
        to_delete: &mut Vec<i32>,
        ret: &mut Vec<Option<Rc<RefCell<TreeNode>>>>,
        is_root: bool,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if root.is_none() {
            return None;
        }
        let is_delete = to_delete.contains(&root.as_ref().unwrap().borrow().val);
        if !is_delete && is_root {
            ret.push(root.clone());
        }
        let (left, right) = (
            root.as_ref().unwrap().borrow().left.clone(),
            root.as_ref().unwrap().borrow().right.clone(),
        );
        root.as_ref().unwrap().borrow_mut().left = dfs(left, to_delete, ret, is_delete);
        root.as_ref().unwrap().borrow_mut().right = dfs(right, to_delete, ret, is_delete);
        if is_delete { None } else { root.clone() }
    }
    dfs(root, &mut to_delete, &mut ret, true);
    ret
}

/// 1111
pub fn max_depth_after_split(seq: String) -> Vec<i32> {
    let (mut d, mut ans) = (0, vec![]);
    seq.chars().for_each(|c| match c {
        '(' => {
            d = d + 1;
            ans.push(d % 2);
        }
        ')' => {
            ans.push(d % 2);
            d = d - 1;
        }
        _ => (),
    });
    ans
}

/// 1114
struct Foo {
    s_1: Sender<()>,
    r_1: Mutex<Receiver<()>>,
    s_2: Sender<()>,
    r_2: Mutex<Receiver<()>>,
}

impl Foo {
    fn new() -> Self {
        let (s_1, r_1) = channel();
        let (s_2, r_2) = channel();
        Foo {
            s_1,
            r_1: Mutex::new(r_1),
            s_2,
            r_2: Mutex::new(r_2),
        }
    }

    fn first<F>(&self, print_first: F)
    where
        F: FnOnce(),
    {
        // Do not change this line
        print_first();
        let _ = self.s_1.send(());
    }

    fn second<F>(&self, print_second: F)
    where
        F: FnOnce(),
    {
        // Do not change this line
        let _ = self.r_1.lock().unwrap().recv();
        print_second();
        let _ = self.s_2.send(());
    }

    fn third<F>(&self, print_third: F)
    where
        F: FnOnce(),
    {
        let _ = self.r_2.lock().unwrap().recv();
        // Do not change this line
        print_third();
    }
}

/// 1115
struct FooBar {
    n: usize,
    state: Mutex<i32>,
    cv: Condvar,
}

impl FooBar {
    fn new(n: usize) -> Self {
        FooBar {
            n,
            state: Mutex::new(0),
            cv: Condvar::new(),
        }
    }

    fn foo<F>(&self, print_foo: F)
    where
        F: Fn(),
    {
        for _ in 0..self.n {
            let mut state = self.state.lock().unwrap();
            while *state != 0 {
                state = self.cv.wait(state).unwrap();
            }
            print_foo();
            *state = 1;
            self.cv.notify_one();
        }
    }

    fn bar<F>(&self, print_bar: F)
    where
        F: Fn(),
    {
        for _ in 0..self.n {
            let mut state = self.state.lock().unwrap();
            while *state != 1 {
                state = self.cv.wait(state).unwrap();
            }
            print_bar();
            *state = 0;
            self.cv.notify_one();
        }
    }
}
