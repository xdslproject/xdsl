(defun my-callback (docstring a b)
  (save-match-data
    (save-excursion
      (forward-whitespace 1)
      (forward-whitespace -1)
      (insert ":")
      (insert (car (last (split-string (car (split-string docstring "\n")) ":")))))
    (if (next-error) (get-type-hint))))

(defun get-type-hint ()
  (interactive)
  (eglot-hover-eldoc-function #'my-callback))
