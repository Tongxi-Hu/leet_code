use std::{cell::RefCell, rc::Rc};

use crate::common::TreeNode;

/// p701
pub fn insert_into_bst(
    root: Option<Rc<RefCell<TreeNode>>>,
    val: i32,
) -> Option<Rc<RefCell<TreeNode>>> {
    let mut root = root;
    if let Some(node) = root.as_ref() {
        let mut tree = node.borrow_mut();
        let v = tree.val;
        if v > val {
            tree.left = insert_into_bst(tree.left.clone(), val);
        } else {
            tree.right = insert_into_bst(tree.right.clone(), val)
        }
    } else {
        root = Some(Rc::new(RefCell::new(TreeNode::new(val))))
    }
    root
}
