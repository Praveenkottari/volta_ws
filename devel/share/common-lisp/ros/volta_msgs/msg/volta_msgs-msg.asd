
(cl:in-package :asdf)

(defsystem "volta_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "BMS" :depends-on ("_package_BMS"))
    (:file "_package_BMS" :depends-on ("_package"))
    (:file "Diagnostics" :depends-on ("_package_Diagnostics"))
    (:file "_package_Diagnostics" :depends-on ("_package"))
    (:file "RPM" :depends-on ("_package_RPM"))
    (:file "_package_RPM" :depends-on ("_package"))
    (:file "Table" :depends-on ("_package_Table"))
    (:file "_package_Table" :depends-on ("_package"))
  ))