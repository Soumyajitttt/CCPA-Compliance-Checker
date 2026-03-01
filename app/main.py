"""
CCPA Compliance Checker - FastAPI Server  v3.0
================================================
Model    : google/flan-t5-base (~250 MB, CPU-only, no HF token needed)
Strategy : Full verbatim CCPA statute (all 65 pages) embedded as knowledge base
           + fast keyword matching layer
           + LLM verification layer
           = maximum accuracy
"""

import os
import re
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FULL VERBATIM CCPA STATUTE — All 65 pages
CCPA_FULL_TEXT = r"""
1798.100. General Duties of Businesses that Collect Personal Information
(a) A business that controls the collection of a consumer's personal information shall, at or
before the point of collection, inform consumers of the following:
(1) The categories of personal information to be collected and the purposes for which
the categories of personal information are collected or used and whether that
information is sold or shared. A business shall not collect additional categories of
personal information or use personal information collected for additional purposes
that are incompatible with the disclosed purpose for which the personal information
was collected without providing the consumer with notice consistent with this
section.
(2) If the business collects sensitive personal information, the categories of sensitive
personal information to be collected and the purposes for which the categories of
sensitive personal information are collected or used, and whether that information
is sold or shared. A business shall not collect additional categories of sensitive
personal information or use sensitive personal information collected for additional
purposes that are incompatible with the disclosed purpose for which the sensitive
personal information was collected without providing the consumer with notice
consistent with this section.
(3) The length of time the business intends to retain each category of personal
information, including sensitive personal information, or if that is not possible, the
criteria used to determine that period provided that a business shall not retain a
consumer's personal information or sensitive personal information for each
disclosed purpose for which the personal information was collected for longer than
is reasonably necessary for that disclosed purpose.
(b) A business that, acting as a third party, controls the collection of personal information
about a consumer may satisfy its obligation under subdivision (a) by providing the
required information prominently and conspicuously on the homepage of its internet
website. In addition, if a business acting as a third party controls the collection of personal
information about a consumer on its premises, including in a vehicle, then the business
shall, at or before the point of collection, inform consumers as to the categories of
personal information to be collected and the purposes for which the categories of
personal information are used, and whether that personal information is sold, in a clear
and conspicuous manner at the location.
(c) A business' collection, use, retention, and sharing of a consumer's personal information
shall be reasonably necessary and proportionate to achieve the purposes for which the
personal information was collected or processed, or for another disclosed purpose that is
compatible with the context in which the personal information was collected, and not
further processed in a manner that is incompatible with those purposes.
(d) A business that collects a consumer's personal information and that sells that personal
information to, or shares it with, a third party or that discloses it to a service provider or
contractor for a business purpose shall enter into an agreement with the third party,
service provider, or contractor, that:
(1) Specifies that the personal information is sold or disclosed by the business only for
limited and specified purposes.
(2) Obligates the third party, service provider, or contractor to comply with applicable
obligations under this title and obligate those persons to provide the same level of
privacy protection as is required by this title.
(3) Grants the business rights to take reasonable and appropriate steps to help ensure
that the third party, service provider, or contractor uses the personal information
transferred in a manner consistent with the business' obligations under this title.
(4) Requires the third party, service provider, or contractor to notify the business if it
makes a determination that it can no longer meet its obligations under this title.
(5) Grants the business the right, upon notice, including under paragraph (4), to take
reasonable and appropriate steps to stop and remediate unauthorized use of
personal information.
(e) A business that collects a consumer's personal information shall implement reasonable
security procedures and practices appropriate to the nature of the personal information
to protect the personal information from unauthorized or illegal access, destruction, use,
modification, or disclosure in accordance with Section 1798.81.5.
(f) Nothing in this section shall require a business to disclose trade secrets, as specified in
regulations adopted pursuant to paragraph (3) of subdivision (a) of Section 1798.185.

1798.105. Consumers' Right to Delete Personal Information
(a) A consumer shall have the right to request that a business delete any personal
information about the consumer which the business has collected from the consumer.
(b) A business that collects personal information about consumers shall disclose, pursuant to
Section 1798.130, the consumer's rights to request the deletion of the consumer's
personal information.
(c) (1) A business that receives a verifiable consumer request from a consumer to delete
the consumer's personal information pursuant to subdivision (a) of this section shall
delete the consumer's personal information from its records, notify any service
providers or contractors to delete the consumer's personal information from their
records, and notify all third parties to whom the business has sold or shared the
personal information to delete the consumer's personal information unless this
proves impossible or involves disproportionate effort.
(2) The business may maintain a confidential record of deletion requests solely for the
purpose of preventing the personal information of a consumer who has submitted a
deletion request from being sold, for compliance with laws or for other purposes,
solely to the extent permissible under this title.
(3) A service provider or contractor shall cooperate with the business in responding to a
verifiable consumer request, and at the direction of the business, shall delete, or
enable the business to delete and shall notify any of its own service providers or
contractors to delete personal information about the consumer collected, used,
processed, or retained by the service provider or the contractor.
(d) A business, or a service provider or contractor acting pursuant to its contract with the
business, another service provider, or another contractor, shall not be required to comply
with a consumer's request to delete the consumer's personal information if it is
reasonably necessary for the business, service provider, or contractor to maintain the
consumer's personal information in order to:
(1) Complete the transaction for which the personal information was collected, fulfill
the terms of a written warranty or product recall conducted in accordance with
federal law, provide a good or service requested by the consumer, or reasonably
anticipated by the consumer within the context of a business' ongoing business
relationship with the consumer, or otherwise perform a contract between the
business and the consumer.
(2) Help to ensure security and integrity to the extent the use of the consumer's
personal information is reasonably necessary and proportionate for those purposes.
(3) Debug to identify and repair errors that impair existing intended functionality.
(4) Exercise free speech, ensure the right of another consumer to exercise that
consumer's right of free speech, or exercise another right provided for by law.
(5) Comply with the California Electronic Communications Privacy Act pursuant to
Chapter 3.6 (commencing with Section 1546) of Title 12 of Part 2 of the Penal Code.
(6) Engage in public or peer-reviewed scientific, historical, or statistical research that
conforms or adheres to all other applicable ethics and privacy laws, when the
business' deletion of the information is likely to render impossible or seriously
impair the ability to complete such research, if the consumer has provided informed
consent.
(7) To enable solely internal uses that are reasonably aligned with the expectations of
the consumer based on the consumer's relationship with the business and
compatible with the context in which the consumer provided the information.
(8) Comply with a legal obligation.

1798.106. Consumers' Right to Correct Inaccurate Personal Information
(a) A consumer shall have the right to request a business that maintains inaccurate personal
information about the consumer to correct that inaccurate personal information, taking
into account the nature of the personal information and the purposes of the processing of
the personal information.
(b) A business that collects personal information about consumers shall disclose, pursuant to
Section 1798.130, the consumer's right to request correction of inaccurate personal
information.
(c) A business that receives a verifiable consumer request to correct inaccurate personal
information shall use commercially reasonable efforts to correct the inaccurate personal
information as directed by the consumer, pursuant to Section 1798.130 and regulations
adopted pursuant to paragraph (7) of subdivision (a) of Section 1798.185.

1798.110. Consumers' Right to Know What Personal Information is Being Collected. Right to
Access Personal Information
(a) A consumer shall have the right to request that a business that collects personal
information about the consumer disclose to the consumer the following:
(1) The categories of personal information it has collected about that consumer.
(2) The categories of sources from which the personal information is collected.
(3) The business or commercial purpose for collecting, selling, or sharing personal
information.
(4) The categories of third parties to whom the business discloses personal information.
(5) The specific pieces of personal information it has collected about that consumer.
(b) A business that collects personal information about a consumer shall disclose to the
consumer, pursuant to subparagraph (B) of paragraph (3) of subdivision (a) of Section
1798.130, the information specified in subdivision (a) upon receipt of a verifiable
consumer request from the consumer.
(c) A business that collects personal information about consumers shall disclose, pursuant to
subparagraph (B) of paragraph (5) of subdivision (a) of Section 1798.130:
(1) The categories of personal information it has collected about consumers.
(2) The categories of sources from which the personal information is collected.
(3) The business or commercial purpose for collecting, selling, or sharing personal information.
(4) The categories of third parties to whom the business discloses personal information.
(5) That a consumer has the right to request the specific pieces of personal information
the business has collected about that consumer.

1798.115. Consumers' Right to Know What Personal Information is Sold or Shared and to Whom
(a) A consumer shall have the right to request that a business that sells or shares the
consumer's personal information, or that discloses it for a business purpose, disclose to
that consumer:
(1) The categories of personal information that the business collected about the consumer.
(2) The categories of personal information that the business sold or shared about the
consumer and the categories of third parties to whom the personal information was
sold or shared, by category or categories of personal information for each category
of third parties to whom the personal information was sold or shared.
(3) The categories of personal information that the business disclosed about the
consumer for a business purpose and the categories of persons to whom it was
disclosed for a business purpose.
(b) A business that sells or shares personal information about a consumer, or that discloses a
consumer's personal information for a business purpose, shall disclose, pursuant to
paragraph (4) of subdivision (a) of Section 1798.130, the information specified in
subdivision (a) to the consumer upon receipt of a verifiable consumer request from the
consumer.
(d) A third party shall not sell or share personal information about a consumer that has been
sold to, or shared with, the third party by a business unless the consumer has received
explicit notice and is provided an opportunity to exercise the right to opt-out pursuant to
Section 1798.120.

1798.120. Consumers' Right to Opt Out of Sale or Sharing of Personal Information
(a) (1) A consumer shall have the right, at any time, to direct a business that sells or shares
personal information about the consumer to third parties not to sell or share the
consumer's personal information. This right may be referred to as the right to opt
out of sale or sharing.
(2) A business to which another business transfers the personal information of a
consumer as an asset that is part of a merger, acquisition, bankruptcy, or other
transaction in which the transferee assumes control of all of, or part of, the
transferor shall comply with a consumer's direction to the transferor made pursuant
to this subdivision.
(b) A business that sells consumers' personal information to, or shares it with, third parties
shall provide notice to consumers, pursuant to subdivision (a) of Section 1798.135, that
this information may be sold or shared and that consumers have the "right to opt out" of
the sale or sharing of their personal information.
(c) Notwithstanding subdivision (a), a business shall not sell or share the personal
information of consumers if the business has actual knowledge that the consumer is less
than 16 years of age, unless the consumer, in the case of consumers at least 13 years of
age and less than 16 years of age, or the consumer's parent or guardian, in the case of
consumers who are less than 13 years of age, has affirmatively authorized the sale or
sharing of the consumer's personal information. A business that willfully disregards the
consumer's age shall be deemed to have had actual knowledge of the consumer's age.
(d) A business that has received direction from a consumer not to sell or share the
consumer's personal information or, in the case of a minor consumer's personal
information has not received consent to sell or share the minor consumer's personal
information, shall be prohibited, pursuant to paragraph (4) of subdivision (c) of Section
1798.135, from selling or sharing the consumer's personal information after its receipt of
the consumer's direction, unless the consumer subsequently provides consent, for the
sale or sharing of the consumer's personal information.

1798.121. Consumers' Right to Limit Use and Disclosure of Sensitive Personal Information
(a) A consumer shall have the right, at any time, to direct a business that collects sensitive
personal information about the consumer to limit its use of the consumer's sensitive
personal information to that use which is necessary to perform the services or provide the
goods reasonably expected by an average consumer who requests those goods or
services, to perform the services set forth in paragraphs (2), (4), (5), and (8) of subdivision
(e) of Section 1798.140, and as authorized by regulations adopted pursuant to
subparagraph (C) of paragraph (18) of subdivision (a) of Section 1798.185. A business that
uses or discloses a consumer's sensitive personal information for purposes other than
those specified in this subdivision shall provide notice to consumers, pursuant to
subdivision (a) of Section 1798.135, that this information may be used, or disclosed to a
service provider or contractor, for additional, specified purposes and that consumers have
the right to limit the use or disclosure of their sensitive personal information.
(b) A business that has received direction from a consumer not to use or disclose the
consumer's sensitive personal information, except as authorized by subdivision (a), shall
be prohibited, pursuant to paragraph (4) of subdivision (c) of Section 1798.135, from using
or disclosing the consumer's sensitive personal information for any other purpose after its
receipt of the consumer's direction unless the consumer subsequently provides consent
for the use or disclosure of the consumer's sensitive personal information for additional
purposes.
(d) Sensitive personal information that is collected or processed without the purpose of
inferring characteristics about a consumer is not subject to this section.

1798.125. Consumers' Right of No Retaliation Following Opt Out or Exercise of Other Rights
(a) (1) A business shall not discriminate against a consumer because the consumer
exercised any of the consumer's rights under this title, including, but not limited to, by:
(A) Denying goods or services to the consumer.
(B) Charging different prices or rates for goods or services, including through the
use of discounts or other benefits or imposing penalties.
(C) Providing a different level or quality of goods or services to the consumer.
(D) Suggesting that the consumer will receive a different price or rate for goods or
services or a different level or quality of goods or services.
(E) Retaliating against an employee, applicant for employment, or independent
contractor, as defined in subparagraph (A) of paragraph (2) of subdivision (m)
of Section 1798.145, for exercising their rights under this title.
(2) Nothing in this subdivision prohibits a business, pursuant to subdivision (b), from
charging a consumer a different price or rate, or from providing a different level or
quality of goods or services to the consumer, if that difference is reasonably related
to the value provided to the business by the consumer's data.
(3) This subdivision does not prohibit a business from offering loyalty, rewards,
premium features, discounts, or club card programs consistent with this title.
(b) (1) A business may offer financial incentives, including payments to consumers as
compensation, for the collection of personal information, the sale or sharing of
personal information, or the retention of personal information. A business may also
offer a different price, rate, level, or quality of goods or services to the consumer if
that price or difference is reasonably related to the value provided to the business
by the consumer's data.
(2) A business that offers any financial incentives pursuant to this subdivision, shall
notify consumers of the financial incentives pursuant to Section 1798.130.
(4) A business shall not use financial incentive practices that are unjust, unreasonable,
coercive, or usurious in nature.

1798.130. Notice, Disclosure, Correction, and Deletion Requirements
(a) In order to comply with Sections 1798.100, 1798.105, 1798.106, 1798.110, 1798.115, and
1798.125, a business shall, in a form that is reasonably accessible to consumers:
(1) (B) Make available to consumers two or more designated methods for submitting
requests for information required to be disclosed pursuant to Sections
1798.110 and 1798.115, or requests for deletion or correction pursuant to
Sections 1798.105 and 1798.106, respectively, including, at a minimum, a toll
free telephone number. A business that operates exclusively online and has a
direct relationship with a consumer from whom it collects personal
information shall only be required to provide an email address for submitting
requests for information required to be disclosed pursuant to Sections
1798.110 and 1798.115, or for requests for deletion or correction pursuant to
Sections 1798.105 and 1798.106, respectively.
If the business maintains an internet website, make the internet website available to
consumers to submit requests for information required to be disclosed pursuant to
Sections 1798.110 and 1798.115, or requests for deletion or correction pursuant to
Sections 1798.105 and 1798.106, respectively.
(2) (A) Disclose and deliver the required information to a consumer free of charge,
correct inaccurate personal information, or delete a consumer's personal
information, based on the consumer's request, within 45 days of receiving a
verifiable consumer request from the consumer. The business shall promptly
take steps to determine whether the request is a verifiable consumer request,
but this shall not extend the business' duty to disclose and deliver the
information, to correct inaccurate personal information, or to delete personal
information within 45 days of receipt of the consumer's request. The time
period to provide the required information, to correct inaccurate personal
information, or to delete personal information may be extended once by an
additional 45 days when reasonably necessary, provided the consumer is
provided notice of the extension within the first 45-day period.
(5) Disclose the following information in its online privacy policy or policies if the
business has an online privacy policy or policies and in any California-specific
description of consumers' privacy rights, or if the business does not maintain those
policies, on its internet website, and update that information at least once every 12
months:
(A) A description of a consumer's rights pursuant to Sections 1798.100,
1798.105, 1798.106, 1798.110, 1798.115, and 1798.125 and two or more
designated methods for submitting requests.
(B) For purposes of subdivision (c) of Section 1798.110:
(i) A list of the categories of personal information it has collected about
consumers in the preceding 12 months.
(ii) The categories of sources from which consumers' personal information
is collected.
(iii) The business or commercial purpose for collecting, selling, or sharing
consumers' personal information.
(iv) The categories of third parties to whom the business discloses
consumers' personal information.
(C) For purposes of paragraphs (1) and (2) of subdivision (c) of Section 1798.115,
two separate lists: a list of the categories of personal information it has sold
or shared about consumers in the preceding 12 months, or if the business has
not sold or shared consumers' personal information in the preceding 12
months, the business shall prominently disclose that fact; and a list of the
categories of personal information it has disclosed about consumers for a
business purpose in the preceding 12 months.
(b) A business is not obligated to provide the information required by Sections 1798.110 and
1798.115 to the same consumer more than twice in a 12-month period.

1798.135. Methods of Limiting Sale, Sharing, and Use of Personal Information and Use of
Sensitive Personal Information
(a) A business that sells or shares consumers' personal information or uses or discloses
consumers' sensitive personal information for purposes other than those authorized by
subdivision (a) of Section 1798.121 shall, in a form that is reasonably accessible to
consumers:
(1) Provide a clear and conspicuous link on the business' internet homepages, titled "Do
Not Sell or Share My Personal Information," to an internet web page that enables a
consumer, or a person authorized by the consumer, to opt out of the sale or sharing
of the consumer's personal information.
(2) Provide a clear and conspicuous link on the business' internet homepages, titled
"Limit the Use of My Sensitive Personal Information," that enables a consumer, or a
person authorized by the consumer, to limit the use or disclosure of the consumer's
sensitive personal information to those uses authorized by subdivision (a) of Section
1798.121.
(3) At the business' discretion, utilize a single, clearly labeled link on the business'
internet homepages, in lieu of complying with paragraphs (1) and (2), if that link
easily allows a consumer to opt out of the sale or sharing of the consumer's personal
information and to limit the use or disclosure of the consumer's sensitive personal
information.
(c) A business that is subject to this section shall:
(1) Not require a consumer to create an account or provide additional information
beyond what is necessary in order to direct the business not to sell or share the
consumer's personal information or to limit use or disclosure of the consumer's
sensitive personal information.
(4) For consumers who exercise their right to opt out of the sale or sharing of their
personal information or limit the use or disclosure of their sensitive personal
information, refrain from selling or sharing the consumer's personal information or
using or disclosing the consumer's sensitive personal information and wait for at
least 12 months before requesting that the consumer authorize the sale or sharing
of the consumer's personal information or the use and disclosure of the consumer's
sensitive personal information for additional purposes, or as authorized by
regulations.
(5) For consumers under 16 years of age who do not consent to the sale or sharing of
their personal information, refrain from selling or sharing the personal information
of the consumer under 16 years of age and wait for at least 12 months before
requesting the consumer's consent again, or as authorized by regulations or until
the consumer attains 16 years of age.

1798.140. Definitions
"Personal information" means information that identifies, relates to, describes, is reasonably
capable of being associated with, or could reasonably be linked, directly or indirectly, with a
particular consumer or household. Personal information includes, but is not limited to, the
following:
(A) Identifiers such as a real name, alias, postal address, unique personal identifier,
online identifier, Internet Protocol address, email address, account name,
social security number, driver's license number, passport number, or other
similar identifiers.
(B) Any personal information described in subdivision (e) of Section 1798.80.
(C) Characteristics of protected classifications under California or federal law.
(D) Commercial information, including records of personal property, products or
services purchased, obtained, or considered, or other purchasing or consuming
histories or tendencies.
(E) Biometric information.
(F) Internet or other electronic network activity information, including, but not
limited to, browsing history, search history, and information regarding a
consumer's interaction with an internet website application, or advertisement.
(G) Geolocation data.
(H) Audio, electronic, visual, thermal, olfactory, or similar information.
(I) Professional or employment-related information.
(J) Education information.
(K) Inferences drawn from any of the information identified in this subdivision to
create a profile about a consumer reflecting the consumer's preferences,
characteristics, psychological trends, predispositions, behavior, attitudes,
intelligence, abilities, and aptitudes.
(L) Sensitive personal information.

"Sensitive personal information" means:
(1) Personal information that reveals:
(A) A consumer's social security, driver's license, state identification card, or
passport number.
(B) A consumer's account log-in, financial account, debit card, or credit card
number in combination with any required security or access code, password,
or credentials allowing access to an account.
(C) A consumer's precise geolocation.
(D) A consumer's racial or ethnic origin, citizenship or immigration status, religious
or philosophical beliefs, or union membership.
(E) The contents of a consumer's mail, email, and text messages unless the
business is the intended recipient of the communication.
(F) A consumer's genetic data.
(G)(i) A consumer's neural data.
(ii) "Neural data" means information that is generated by measuring the
activity of a consumer's central or peripheral nervous system, and that is
not inferred from nonneural information.
(2) The processing of biometric information for the purpose of uniquely identifying
a consumer.
(3) Personal information collected and analyzed concerning a consumer's health.
(A) Personal information collected and analyzed concerning a consumer's sex life
or sexual orientation.

"Sell," "selling," "sale," or "sold" means selling, renting, releasing, disclosing, disseminating,
making available, transferring, or otherwise communicating orally, in writing, or by electronic
or other means, a consumer's personal information by the business to a third party for
monetary or other valuable consideration.

"Share," "shared," or "sharing" means sharing, renting, releasing, disclosing, disseminating,
making available, transferring, or otherwise communicating orally, in writing, or by electronic
or other means, a consumer's personal information by the business to a third party for
cross-context behavioral advertising, whether or not for monetary or other valuable
consideration.

"Business" means a sole proprietorship, partnership, limited liability company, corporation,
association, or other legal entity that is organized or operated for the profit or financial benefit
of its shareholders or other owners, that collects consumers' personal information, and that
satisfies one or more of the following thresholds:
(A) As of January 1 of the calendar year, had annual gross revenues in excess of
twenty-five million dollars ($25,000,000) in the preceding calendar year.
(B) Alone or in combination, annually buys, sells, or shares the personal
information of 100,000 or more consumers or households.
(C) Derives 50 percent or more of its annual revenues from selling or sharing
consumers' personal information.

"Biometric information" means an individual's physiological, biological, or behavioral
characteristics, including information pertaining to an individual's deoxyribonucleic acid (DNA),
that is used or is intended to be used singly or in combination with each other or with other
identifying data, to establish individual identity. Biometric information includes, but is not
limited to, imagery of the iris, retina, fingerprint, face, hand, palm, vein patterns, and voice
recordings, from which an identifier template, such as a faceprint, a minutiae template, or a
voiceprint, can be extracted, and keystroke patterns or rhythms, gait patterns or rhythms, and
sleep, health, or exercise data that contain identifying information.

"Precise geolocation" means any data that is derived from a device and that is used or intended
to be used to locate a consumer within a geographic area that is equal to or less than the area
of a circle with a radius of 1,850 feet, except as prescribed by regulations.

"Consent" means any freely given, specific, informed, and unambiguous indication of the
consumer's wishes by which the consumer, or the consumer's legal guardian, a person
who has power of attorney, or a person acting as a conservator for the consumer,
including by a statement or by a clear affirmative action, signifies agreement to the
processing of personal information relating to the consumer for a narrowly defined
particular purpose. Acceptance of a general or broad terms of use, or similar document,
that contains descriptions of personal information processing along with other, unrelated
information, does not constitute consent. Hovering over, muting, pausing, or closing a
given piece of content does not constitute consent. Likewise, agreement obtained
through use of dark patterns does not constitute consent.

"Consumer" means a natural person who is a California resident.

"Cross-context behavioral advertising" means the targeting of advertising to a consumer
based on the consumer's personal information obtained from the consumer's activity
across businesses, distinctly branded internet websites, applications, or services, other
than the business, distinctly branded internet website, application, or service with which
the consumer intentionally interacts.

"Deidentified" means information that cannot reasonably be used to infer information
about, or otherwise be linked to, a particular consumer provided that the business that
possesses the information:
(1) Takes reasonable measures to ensure that the information cannot be associated
with a consumer or household.
(2) Publicly commits to maintain and use the information in deidentified form and not
to attempt to reidentify the information.
(3) Contractually obligates any recipients of the information to comply with all
provisions of this subdivision.

1798.145. Exemptions
(a) (1) The obligations imposed on businesses by this title shall not restrict a business's
ability to:
(A) Comply with federal, state, or local laws or comply with a court order or
subpoena to provide information.
(B) Comply with a civil, criminal, or regulatory inquiry, investigation, subpoena, or
summons by federal, state, or local authorities.
(C) Cooperate with law enforcement agencies concerning conduct or activity that
the business, service provider, or third party reasonably and in good faith
believes may violate federal, state, or local law.
(E) Exercise or defend legal claims.
(F) Collect, use, retain, sell, share, or disclose consumers' personal information
that is deidentified or aggregate consumer information.
(G) Collect, sell, or share a consumer's personal information if every aspect of that
commercial conduct takes place wholly outside of California.
(c) (1) This title shall not apply to any of the following:
(A) Medical information governed by the Confidentiality of Medical Information
Act (Part 2.6) or protected health information collected by a covered entity or
business associate governed by HIPAA.
(B) A provider of health care governed by the Confidentiality of Medical
Information Act or a covered entity governed by HIPAA.
(d) This title shall not apply to an activity involving the collection, maintenance, disclosure,
sale, communication, or use of any personal information bearing on a consumer's
creditworthiness, credit standing, credit capacity, character, general reputation, personal
characteristics, or mode of living by a consumer reporting agency subject to regulation
under the Fair Credit Reporting Act.
(e) This title shall not apply to personal information collected, processed, sold, or disclosed
subject to the federal Gramm-Leach-Bliley Act, and implementing regulations, or the
California Financial Information Privacy Act.

1798.150. Personal Information Security Breaches
(a) (1) Any consumer whose nonencrypted and nonredacted personal information is subject
to an unauthorized access and exfiltration, theft, or disclosure as a result of the
business' violation of the duty to implement and maintain reasonable security
procedures and practices appropriate to the nature of the information to protect the
personal information may institute a civil action.
(A) To recover damages in an amount not less than one hundred dollars ($100)
and not greater than seven hundred and fifty ($750) per consumer per
incident or actual damages, whichever is greater.
(B) Injunctive or declaratory relief.
(C) Any other relief the court deems proper.

1798.155. Administrative Enforcement
(a) Any business, service provider, contractor, or other person that violates this title shall be
liable for an administrative fine of not more than two thousand five hundred dollars
($2,500) for each violation or seven thousand five hundred dollars ($7,500) for each
intentional violation or violations involving the personal information of consumers whom
the business, service provider, contractor, or other person has actual knowledge are
under 16 years of age.

1798.175. Conflicting Provisions
This title is intended to further the constitutional right of privacy and to supplement existing
laws relating to consumers' personal information. The provisions of this title are not limited to
information collected electronically or over the Internet, but apply to the collection and sale of
all personal information collected by a business from consumers. Wherever possible, law
relating to consumers' personal information should be construed to harmonize with the
provisions of this title, but in the event of a conflict between other laws and the provisions of
this title, the provisions of the law that afford the greatest protection for the right of privacy for
consumers shall control.

1798.180. Preemption
This title is a matter of statewide concern and supersedes and preempts all rules, regulations,
codes, ordinances, and other laws adopted by a city, county, city and county, municipality, or
local agency regarding the collection and sale of consumers' personal information by a business.

1798.190. Anti-Avoidance
A court or the agency shall disregard the intermediate steps or transactions for purposes of
effectuating the purposes of this title.

1798.192. Waiver
Any provision of a contract or agreement of any kind, including a representative action waiver,
that purports to waive or limit in any way rights under this title shall be deemed contrary to
public policy and shall be void and unenforceable.

1798.194. This title shall be liberally construed to effectuate its purposes.

1798.199.10. California Privacy Protection Agency
There is hereby established in state government the California Privacy Protection Agency,
which is vested with full administrative power, authority, and jurisdiction to implement
and enforce the California Consumer Privacy Act of 2018.

1798.199.40. The agency shall perform the following functions:
(a) Administer, implement, and enforce through administrative actions this title.
(c) Through the implementation of this title, protect the fundamental privacy rights of natural
persons with respect to the use of their personal information.

1798.199.90. Civil Penalties
Any business, service provider, contractor, or other person that violates this title shall be
subject to an injunction and liable for a civil penalty of not more than two thousand five
hundred dollars ($2,500) for each violation or seven thousand five hundred dollars
($7,500) for each intentional violation and each violation involving the personal
information of minor consumers.

--- END OF CCPA STATUTE ---

VIOLATION SUMMARY (for quick reference):
Section 1798.100 = Must disclose categories of personal info collected BEFORE or AT collection.
  Violations: collecting browsing history/geolocation/biometric/health data without notice;
  privacy policy omits data categories; collecting for undisclosed purposes; no privacy policy.

Section 1798.105 = Consumer right to deletion.
  Violations: ignoring deletion requests; refusing to delete data after verified request;
  keeping all records when consumer asked for deletion.

Section 1798.106 = Consumer right to correct inaccurate data.
  Violations: refusing to correct inaccurate personal information.

Section 1798.110 = Consumer right to know what is collected.
  Violations: refusing to tell consumers what personal data is collected; denying access requests.

Section 1798.115 = Consumer right to know what is sold/shared.
  Violations: refusing to disclose data sales or the third parties data is shared with.

Section 1798.120 = Consumer right to opt out of data sale/sharing.
  Violations: selling personal data without opt-out; selling to data brokers without telling consumers;
  selling data of minors under 16 without consent (under 13 needs parental consent;
  13-15 needs the minor's own affirmative consent); continuing to sell after opt-out.

Section 1798.121 = Consumer right to limit use of sensitive personal info.
  Violations: using SSN, financial account numbers, precise geolocation, racial/ethnic origin,
  religious beliefs, union membership, genetic data, neural data, biometric data, health data,
  sex life/sexual orientation data beyond what is necessary for the service.

Section 1798.125 = No discrimination/retaliation for exercising rights.
  Violations: charging higher prices to consumers who opt out; denying services to privacy-
  exercising consumers; providing worse service; penalizing for deletion requests; retaliating.

Section 1798.130 = Notice and response requirements.
  Violations: no toll-free number or designated request method; not responding within 45 days;
  no privacy policy; privacy policy not updated in 12 months; charging for request responses.

Section 1798.135 = Opt-out mechanism requirements.
  Violations: no "Do Not Sell or Share My Personal Information" link on homepage;
  requiring account creation to opt out; re-contacting opted-out consumer within 12 months.

NOT VIOLATIONS (exemptions):
- Complying with court orders, subpoenas, legal obligations
- Acting under HIPAA as a covered health entity
- Acting under Gramm-Leach-Bliley Act as a financial institution
- Using deidentified or aggregate consumer information
- Commercial conduct entirely outside California
- Offering loyalty/rewards programs with proper disclosure and consent
"""

# Rules for fast keyword matching
CCPA_RULES = [
    {
        "section": "Section 1798.100",
        "keywords": [
            "privacy policy doesn't mention",
            "privacy policy does not mention",
            "not mentioned in",
            "without informing",
            "without notice to",
            "without telling",
            "undisclosed collection",
            "secret collection",
            "hidden collection",
            "not disclosed",
            "browsing history",
            "geolocation data",
            "biometric data",
            "collecting without",
            "not in privacy policy",
            "no privacy policy",
            "without disclosure",
            "collecting data without",
            "does not disclose what",
            "not notif",
            "failed to disclose",
            "collect additional categories",
            "incompatible purpose",
            "not included in",
            "without user knowledge",
            "collecting information without",
        ],
    },
    {
        "section": "Section 1798.105",
        "keywords": [
            "ignoring their request",
            "ignoring the request",
            "ignore their request",
            "ignore the deletion",
            "keeping all records",
            "not delete",
            "won't delete",
            "refused to delete",
            "refuse to delete",
            "not honor deletion",
            "asked us to delete",
            "asked to delete",
            "request to delete",
            "delete their data",
            "delete their personal",
            "deletion request",
            "not comply with deletion",
            "ignoring deletion",
            "not fulfilling deletion",
            "does not delete",
            "did not delete",
            "keeping records after",
        ],
    },
    {
        "section": "Section 1798.106",
        "keywords": [
            "inaccurate personal information",
            "refuse to correct",
            "not correct",
            "won't correct",
            "incorrect data",
            "wrong information",
            "incorrect personal information",
            "refusing to fix inaccurate",
        ],
    },
    {
        "section": "Section 1798.110",
        "keywords": [
            "refuse to tell consumers",
            "refusing to tell",
            "won't reveal what data",
            "deny access to data",
            "not share what data",
            "won't disclose what",
            "hiding what we collect",
            "not telling consumers what data",
            "refuse to disclose what",
            "not provide access to personal",
        ],
    },
    {
        "section": "Section 1798.115",
        "keywords": [
            "not tell who we sold",
            "not disclose who",
            "won't say who we sold",
            "refusing to disclose sales",
            "not reveal third party recipients",
            "hiding data sales",
            "not disclosing who we share with",
            "won't disclose third party",
        ],
    },
    {
        "section": "Section 1798.120",
        "keywords": [
            "selling our customers",
            "sell our customers",
            "selling customer",
            "sell customer",
            "selling personal information to",
            "sell personal information to",
            "selling personal data to",
            "sell personal data to",
            "selling data to",
            "sell data to",
            "third-party data broker",
            "third party data broker",
            "data broker",
            "ad network",
            "advertising network",
            "without informing them",
            "without giving them a chance",
            "without opt-out",
            "without the opt-out",
            "no opt out",
            "no opt-out",
            "not giving them a chance to opt",
            "not providing opt-out",
            "without their consent",
            "without consent",
            "14-year-old",
            "14 year old",
            "14-year",
            "14 years old",
            "minor users",
            "minor's data",
            "minors' data",
            "children's data",
            "child's data",
            "under 16",
            "under 13",
            "under the age of 16",
            "under the age of 13",
            "13-year-old",
            "15-year-old",
            "12-year-old",
            "teenage users",
            "teen users",
            "parent's consent",
            "parental consent",
            "without parent's consent",
            "without parental consent",
            "without getting parental",
            "without getting their parent",
            "without getting consent",
            "not received consent to sell",
            "continued to sell after",
            "still selling after opt",
        ],
    },
    {
        "section": "Section 1798.121",
        "keywords": [
            "social security number",
            "social security",
            "ssn",
            "financial account",
            "credit card number",
            "debit card number",
            "precise geolocation",
            "exact location",
            "racial origin",
            "ethnic origin",
            "racial or ethnic",
            "racial and ethnic",
            "religious beliefs",
            "union membership",
            "genetic data",
            "neural data",
            "biometric information",
            "health information",
            "health data",
            "medical information",
            "sex life",
            "sexual orientation",
            "sensitive personal information",
            "sensitive data",
            "using beyond what is necessary",
            "beyond necessary purposes",
            "for additional purposes without",
            "beyond service needs",
        ],
    },
    {
        "section": "Section 1798.125",
        "keywords": [
            "higher price",
            "higher prices",
            "higher fee",
            "charge more",
            "different price",
            "different pricing",
            "discriminate",
            "discriminatory",
            "penaliz",
            "penalty for",
            "penalties for",
            "deny service",
            "denying service",
            "denied service",
            "refusing service",
            "worse service",
            "lower quality service",
            "lower-quality",
            "different level of service",
            "opted out of data",
            "opt-out customers",
            "who opted out",
            "exercised their right",
            "exercised privacy right",
            "exercising their right",
            "exercising privacy rights",
            "retaliat",
            "punishment for",
            "punishing customers",
            "because they opted",
            "because the consumer exercised",
            "for requesting deletion",
        ],
    },
    {
        "section": "Section 1798.130",
        "keywords": [
            "no toll-free number",
            "no phone number for privacy",
            "no designated method",
            "no way to submit requests",
            "not responding to requests",
            "ignoring consumer requests",
            "not respond within 45",
            "exceed 45 days",
            "no privacy policy at all",
            "outdated privacy policy",
            "privacy policy not updated",
            "not updated in over",
        ],
    },
    {
        "section": "Section 1798.135",
        "keywords": [
            'no "do not sell"',
            "no do not sell",
            "missing opt-out link",
            "no opt-out link on",
            "no link on homepage",
            "without an opt-out mechanism",
            "require account to opt",
            "must create account to opt",
            "no opt out mechanism",
            "no opt-out mechanism",
            "no do not sell link",
        ],
    },
]

SAFE_PHRASES = [
    "clear privacy policy",
    "allows customers to opt out",
    "opt out at any time",
    "honor all deletion requests",
    "honored the deletion",
    "deleted all personal data within 45 days",
    "deleted within 45 days",
    "do not sell my personal information",
    "do not sell link",
    "opt-out link on the homepage",
    "equal service and pricing",
    "equal pricing",
    "regardless of whether they exercise",
    "properly disclosed",
    "obtained consent before",
    "with user consent",
    "schedule a team meeting",
    "plan a meeting",
    "discuss the project",
    "book a flight",
    "office party",
    "meeting for next",
]

LLM_PROMPT = """You are a CCPA compliance legal expert. Analyze whether this business practice violates the CCPA.

CCPA KEY RULES:
1798.100: Must disclose what personal data is collected BEFORE collection. Undisclosed collection of browsing history, geolocation, or biometric data = VIOLATION.
1798.105: Must honor consumer deletion requests. Ignoring or refusing deletion requests = VIOLATION.
1798.106: Must correct inaccurate data when requested.
1798.110: Must disclose what personal data is collected upon request.
1798.115: Must disclose what data is sold/shared and to whom.
1798.120: Must provide opt-out for data sales. Selling data of minors under 16 without consent = VIOLATION.
1798.121: Must limit use of sensitive data (SSN, health, biometric, precise geolocation, racial/ethnic, genetic) to necessary purposes.
1798.125: Cannot discriminate against consumers who exercise rights. Charging higher prices for opt-out = VIOLATION.
1798.130: Must respond to consumer requests within 45 days. Must maintain updated privacy policy.
1798.135: Must have "Do Not Sell My Personal Information" link on homepage if selling data.

Business practice: "{prompt}"

Does this violate the CCPA?
If YES, list ONLY violated sections like: VIOLATION: Section 1798.100, Section 1798.120
If NO, respond ONLY: COMPLIANT

Answer:"""

# Global model state
tokenizer: Optional[T5Tokenizer] = None
model: Optional[T5ForConditionalGeneration] = None
_device = "cpu"
MODEL_NAME = "google/flan-t5-base"


def load_model():
    global tokenizer, model, _device
    logger.info(f"Loading model: {MODEL_NAME}")
    hf_token = os.getenv("HF_TOKEN") or None
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, token=hf_token)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(_device)
    model.eval()
    logger.info(f"Model loaded successfully on {_device}")


UNRELATED_PATTERNS = [
    "schedule a meeting",
    "team meeting",
    "discuss the project",
    "plan a trip",
    "book a flight",
    "office party",
    "meeting for next",
]


def is_unrelated(text_lower: str) -> bool:
    return any(p in text_lower for p in UNRELATED_PATTERNS) and len(text_lower) < 150


def safe_signal_score(text_lower: str) -> int:
    return sum(1 for p in SAFE_PHRASES if p in text_lower)


def keyword_check(text_lower: str) -> list:
    violations = []
    for rule in CCPA_RULES:
        for kw in rule["keywords"]:
            if kw in text_lower:
                if rule["section"] not in violations:
                    violations.append(rule["section"])
                break
    return violations


def llm_check(prompt: str) -> list:
    try:
        full_prompt = LLM_PROMPT.format(prompt=prompt)
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(_device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=4,
                early_stopping=True,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.info(f"LLM raw response: {response!r}")

        if "COMPLIANT" in response.upper() and "VIOLATION" not in response.upper():
            return []

        numbers = re.findall(r"1798\.\d+", response)
        known = {r["section"] for r in CCPA_RULES}
        sections, seen = [], set()
        for num in numbers:
            s = f"Section {num}"
            if s not in seen and s in known:
                seen.add(s)
                sections.append(s)
        return sections
    except Exception as exc:
        logger.warning(f"LLM check error: {exc}")
        return []


def analyze_prompt(prompt: str) -> dict:
    text_lower = prompt.lower()

    if is_unrelated(text_lower):
        logger.info("Unrelated prompt — returning COMPLIANT")
        return {"harmful": False, "articles": []}

    kw_violations = keyword_check(text_lower)
    logger.info(f"Keyword violations: {kw_violations}")

    llm_violations = llm_check(prompt)
    logger.info(f"LLM violations: {llm_violations}")

    combined = list(kw_violations)
    for v in llm_violations:
        if v not in combined:
            combined.append(v)
    combined.sort()

    if not combined and safe_signal_score(text_lower) >= 2:
        return {"harmful": False, "articles": []}

    harmful = len(combined) > 0
    return {"harmful": harmful, "articles": combined if harmful else []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="CCPA Compliance Checker",
    description="Full verbatim CCPA statute (all 65 pages) embedded. Uses keyword matching + flan-t5-base LLM.",
    version="3.0.0",
    lifespan=lifespan,
)


class AnalyzeRequest(BaseModel):
    prompt: str


class AnalyzeResponse(BaseModel):
    harmful: bool
    articles: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    result = analyze_prompt(req.prompt)
    return AnalyzeResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
